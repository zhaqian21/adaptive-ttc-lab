"""Run paper-protocol E2–E4 against a local vLLM engine on one H20."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Lock, local
import time
from unittest.mock import patch
from urllib.parse import urlparse

import requests
import torch
from transformers import AutoTokenizer
import yaml

import run_smoke as runner
from audit_results import audit, check_problem
from paper_protocol import PaperAgent, PaperPRM, PaperSearch
from BALROG.balrog.client import TransformerWrapper


context = local()
PRE_FINAL_FALLBACK_SHA256 = "8df7bdb5997d099e84d2fcb881ee54f76e12dc944286dc1385488b6cc5678692"


class ModelProxy:
    """Supply token IDs from vLLM to the upstream TransformerWrapper."""

    device = torch.device("cpu")

    def __init__(self, endpoint, model_name, decode, seed, session):
        self.endpoint = endpoint
        self.model_name = model_name
        self.decode = decode
        self.seed = seed
        self.session = session
        self.calls = []
        self.method = "generate"

    def generate(self, *, input_ids, max_new_tokens, eos_token_id, **options):
        prompt, = input_ids.tolist()
        assert options["do_sample"] and options["temperature"] == self.decode["temperature"]
        assert options["top_p"] == self.decode["top_p"] and max_new_tokens == self.decode["max_tokens"]
        assert self.decode["stop_token_ids"] == [eos_token_id]
        assert len(prompt) + max_new_tokens <= 32768, "Prompt exceeds the engine context"
        seed = self.seed + 500 * len(self.calls)
        began = time.perf_counter()
        reply = self.session.post(
            self.endpoint + "/v1/completions",
            json={**self.decode, "model": self.model_name, "prompt": prompt,
                  "seed": seed, "return_token_ids": True}, timeout=(5, 240),
        )
        reply.raise_for_status()
        received = reply.json()
        choice, = received["choices"]
        output = choice["token_ids"]
        assert choice["prompt_token_ids"] == prompt
        assert len(output) == received["usage"]["completion_tokens"] and 0 < len(output) <= max_new_tokens
        self.calls.append({"method": self.method, "seconds": time.perf_counter() - began,
                           "seed": seed, "input_tokens": len(prompt), "output_tokens": len(output)})
        return torch.tensor([prompt + output], dtype=input_ids.dtype)


class VllmClient(TransformerWrapper):
    def __init__(self, settings, endpoint, model_name, decode, seed):
        super().__init__(settings)
        self._initialized = True
        if not hasattr(context, "tokenizer"):
            context.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True, trust_remote_code=False)
            context.session = requests.Session()
            context.session.trust_env = False
        self.tokenizer = context.tokenizer
        self.model = ModelProxy(endpoint, model_name, decode, seed, context.session)

    def generate_with_structured(self, messages, schema):
        self.model.method = "generate_with_structured"
        try:
            for _ in range(3):
                response = super().generate_with_structured(messages, schema)
                try:
                    if isinstance(json.loads(response.completion), dict):
                        return response
                except json.JSONDecodeError:
                    pass
            if "answer" in schema.get("properties", {}):
                # The upstream final-answer step can extract an auxiliary answer
                # from raw text. Grading reads the separate boxed completion.
                return response
            raise ValueError(f"Invalid structured output after three generations: {response.completion!r}")
        finally:
            self.model.method = "generate"


class ConcurrentPRM(PaperPRM):
    def __init__(self):
        super().__init__(str(runner.asset_path(runner.ASSETS[2])), device="cuda:0")
        self.lock = Lock()
        runner.check_placement(self.model)

    def _record(self, method, records, *args):
        start = time.perf_counter()
        with self.lock:
            scores = method(*args)
        records.append({"seconds": time.perf_counter() - start, **scores})
        return scores

    def score_reasoning(self, *args):
        return self._record(super().score_reasoning, context.observation["prm_calls"], *args)

    def score_search(self, *args):
        return self._record(super().score_search, context.observation["intra_iteration_scoring"], *args)


def strategy_factory(prm):
    def create(strategy_name, _legacy_scorer, rng=None):
        strategy = PaperSearch(strategy_name, prm, rng)
        observation = context.observation
        calls, scores = observation["llm_calls"], observation["intra_iteration_scoring"]
        generated, scored = len(calls), len(scores)

        def observe(_result, seconds):
            batch = calls[generated:]
            observation["strategy_calls"].append({"strategy": strategy_name, "seconds": seconds,
                                                  "llm_calls": len(batch), "local_scoring_calls": len(scores) - scored,
                                                  "input_tokens": sum(call["input_tokens"] for call in batch),
                                                  "output_tokens": sum(call["output_tokens"] for call in batch)})

        runner.record_call(strategy, "execute", observe)
        return strategy

    return create


def solve_question(question, config, output, endpoint, served_model, decode):
    index = question["problem_id"]
    destination = output / "problems" / f"{index:03d}"
    if (destination / "result.json").is_file():
        return False
    attempt = Path(tempfile.mkdtemp(prefix=f"{index:03d}-", dir=output / "attempts"))
    seed = config["reproduction"]["seed"] + index
    context.client = VllmClient(runner.upstream.DictConfig(config["client"]), endpoint, served_model, decode, seed)
    context.observation = {"seed": seed, "classification": config["reproduction"]["classification"],
                           "backend": "vllm", "llm_calls": context.client.model.calls,
                           "prm_calls": [], "intra_iteration_scoring": [], "traces": [], "strategy_calls": []}
    observation = context.observation
    began = time.perf_counter()
    try:
        agent = runner.upstream.create_universal_agent(config)
        core = runner.math_core.MATHCore(agent, runner.upstream.get_prompts())

        def record_trace(response, _seconds):
            observation["traces"].append({"structured_answer": response.answer,
                                          "compute_strategy_outputs": response.compute_strategy_outputs,
                                          "reasoner_outputs": response.reasoner_outputs})

        runner.record_call(agent, "solve", record_trace)
        result, = runner.upstream.evaluate_reasoning_agent(
            agent_factory=lambda: core, dataset=[question], output_dir=str(attempt),
            agent_config=config["agent"], check_answer_fn=core.check_answer,
            extract_prediction_fn=core.extract_prediction, csv_filename_template="results_live.csv",
            json_filename_template="results_final.json", pathway_filename_template="pathway.json", prompt_kwargs={},
            num_iterations=config["eval"]["num_iterations"], use_prm_selection=config["eval"]["use_prm_selection"],
            prm_selection_metric="mean_reward", verbose=False, dataset_name="MATH",
            extra_problem_keys=["dataset_unique_id", "gold_answer_raw", "gold_answer_normalized", "solution"],
        )["results"]
        observation.update(seconds=time.perf_counter() - began, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result["observation"] = observation
        result_file = attempt / "result.json"
        runner.write_json(result_file, result)
        saved = json.loads(result_file.read_text(encoding="utf-8"))
        check_problem(attempt, saved, index, config["eval"]["num_iterations"], observation["classification"],
                      direct=config["reproduction"]["case"] == "E2")
        attempt.rename(destination)
        return True
    except Exception as error:
        runner.write_json(attempt / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                    "observation": observation})
        raise


def summarize(output, config, questions, wall_seconds):
    rows = [json.loads((output / "problems" / f"{i:03d}" / "result.json").read_text()) for i in range(len(questions))]
    observations = [row["observation"] for row in rows]
    calls = [call for row in observations for call in row["llm_calls"]]
    final_scores = [call for row in observations for call in row["prm_calls"]]
    local_scores = [call for row in observations for call in row["intra_iteration_scoring"]]
    correct = sum(bool(row["final_is_correct"]) for row in rows)
    seconds = sum(row["seconds"] for row in observations)
    reproduction = config["reproduction"]
    summary = {"case": reproduction["case"], "classification": reproduction["classification"],
               "label": reproduction["label"], "backend": "vllm", "samples": len(rows), "correct": correct,
               "accuracy": correct / len(rows), "wall_seconds_last_attempt": wall_seconds,
               "problem_wall_seconds": seconds, "mean_seconds_per_question": seconds / len(rows),
               "gpu_peak_allocated_bytes": max(row["gpu_peak_allocated_bytes"] for row in observations),
               "gpu_peak_reserved_bytes": max(row["gpu_peak_reserved_bytes"] for row in observations),
               "gpu_peak_scope": "shared PRM process only; vLLM memory is recorded separately by nvidia-smi",
               "llm_calls": len(calls), "llm_seconds": sum(call["seconds"] for call in calls),
               "input_tokens": sum(call["input_tokens"] for call in calls),
               "output_tokens": sum(call["output_tokens"] for call in calls),
               "prm_calls": len(final_scores), "prm_seconds": sum(call["seconds"] for call in final_scores),
               "intra_iteration_scoring_calls": len(local_scores),
               "intra_iteration_scoring_seconds": sum(call["seconds"] for call in local_scores)}
    with tempfile.TemporaryDirectory(dir=output) as work:
        pending = Path(work) / "summary.json"
        runner.write_json(pending, summary)
        pending.rename(output / "summary.json")
    return summary


def save_config(output, config):
    path = output / "config.yaml"
    serialized = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous == serialized:
            return
        # The precise prior adapter differs only on final structured generations
        # that exhausted all attempts. It committed no checkpoint for those jobs.
        import copy

        compatible = copy.deepcopy(config)
        compatible["reproduction"].pop("structured_final_fallback")
        compatible["reproduction"]["runner_sha256"] = PRE_FINAL_FALLBACK_SHA256
        canonical = yaml.safe_load(yaml.safe_dump(compatible, sort_keys=False))
        if yaml.safe_load(previous) != canonical:
            raise RuntimeError("Resume requires the same configuration")
        original = output / "config.pre_final_fallback.yaml"
        if original.exists():
            assert original.read_text(encoding="utf-8") == previous
        else:
            with original.open("x", encoding="utf-8") as stream:
                stream.write(previous)
        pending = path.with_suffix(".tmp")
        pending.write_text(serialized, encoding="utf-8")
        pending.replace(path)
        print("Resuming with the auxiliary final-answer fallback; verified checkpoints are retained.", flush=True)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)


def run(args):
    endpoint = urlparse(args.endpoint)
    assert endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1" and endpoint.path == ""
    e1 = json.loads(args.e1_config.read_text(encoding="utf-8"))
    config = runner.configure(args.case, e1["requests"][0]["seed"], args.limit)
    count = config["eval"]["max_problems"]
    dataset_path = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert dataset_hash == e1["dataset_sha256"] and config["client"]["model_id"] == e1["model"]
    assert config["reproduction"]["source_sha256"] == e1["source_sha256"]
    runtime = runner.require_h20()
    assert runtime["cuda_visible_devices"] == e1["gpu_uuid"]
    config["reproduction"].update(backend="vllm", runtime=runtime, dataset_sha256=dataset_hash,
                                  decode=e1["decode"], num_workers=args.workers, vllm_version=args.vllm_version,
                                  endpoint=args.endpoint, served_model=args.served_model,
                                  request_seed_rule="base seed + MATH-500 test row index + 500 * generation index",
                                  structured_retries="up to two new samples for invalid JSON objects",
                                  structured_final_fallback="after all JSON attempts, use the upstream auxiliary final-answer extractor",
                                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("problems", "attempts"):
            (output / name).mkdir(exist_ok=True)
        save_config(output, config)
        questions = runner.dataset(count)
        prm = ConcurrentPRM()
        began = time.perf_counter()
        failures = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner.upstream, "UniversalAgent", PaperAgent))
            stack.enter_context(patch.object(runner.upstream, "create_llm_client", lambda _settings: context.client))
            stack.enter_context(patch.object(runner.upstream, "PRMSelector", lambda: prm))
            stack.enter_context(patch.object(runner.universal_agent, "get_compute_strategy", strategy_factory(prm)))
            workers = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
            jobs = {workers.submit(solve_question, row, config, output, args.endpoint, args.served_model, e1["decode"]): row
                    for row in questions if not (output / "problems" / f"{row['problem_id']:03d}" / "result.json").exists()}
            done = count - len(jobs)
            for job in as_completed(jobs):
                try:
                    done += int(job.result())
                    print(f"{args.case}: {done}/{count} verified checkpoints", flush=True)
                except Exception as error:
                    problem_id = jobs[job]["problem_id"]
                    failures.append(problem_id)
                    print(f"{args.case} problem {problem_id} failed: {error}", flush=True)
                    if len(failures) >= args.workers:
                        for pending in jobs:
                            pending.cancel()
                        break
        if failures:
            raise RuntimeError(f"{args.case} has {len(failures)} failed questions; resume retries only missing checkpoints")
        summary = summarize(output, config, questions, time.perf_counter() - began)
        canonical = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        evidence = audit(output, canonical)
        runner.write_json(output / "audit.json", {**evidence, "backend": "vllm"})
        print(f"{args.case}: {summary['correct']}/{count} correct after audit.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("E2", "E3", "E4"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--e1-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())

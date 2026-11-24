"""Benchmark or run the pinned public MATH-500 code with native or batched decoding."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock, local
import time
from unittest.mock import patch

import requests
import torch
from transformers import AutoTokenizer, GenerationConfig

import run_smoke as runner
from audit_results import check_problem
from BALROG.balrog.client import TransformerWrapper


CASES = {"E1": (False, 1), "E2": (False, 10), "E3": (True, 1), "E4": (True, 10)}
THREAD = local()
CLASSIFICATION = "published_code"
DATASET_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def configuration(case):
    dynamic, iterations = CASES[case]
    upstream_config = runner.ROOT / "upstream" / "config.yaml"
    config = runner.upstream.load_config(str(upstream_config))
    config["agent"].update(mode="dynamic" if dynamic else "direct", use_planner=dynamic,
                           use_tool_selector=dynamic, use_compute_selector=dynamic,
                           fixed_tool=None, fixed_compute=None)
    config["client"].update(client_name="transformer", base_url=None,
                            model_id=str(runner.asset_path(runner.ASSETS[1])),
                            device_map="cuda:0", torch_dtype="bfloat16", trust_remote_code=False)
    config["eval"].update(max_problems=500, num_iterations=iterations, use_prm_selection=iterations > 1,
                          prm_selection_metric="mean_reward", split="test", output_dir=None)
    return config


def effective_decode(config, tokenizer):
    args = config["client"]["generate_kwargs"]
    model_config = GenerationConfig.from_pretrained(config["client"]["model_id"], local_files_only=True)
    assert args["temperature"] > 0 and model_config.num_beams == 1 and model_config.min_new_tokens is None
    return {
        "temperature": args["temperature"], "top_p": args.get("top_p", model_config.top_p),
        "top_k": args.get("top_k", model_config.top_k), "max_tokens": args["max_tokens"],
        "repetition_penalty": model_config.repetition_penalty,
        "stop_token_ids": [tokenizer.eos_token_id], "ignore_eos": True, "skip_special_tokens": True,
    }


class LockedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.lock = Lock()

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        with self.lock:
            return self.tokenizer.apply_chat_template(*args, **kwargs)


class GenerationProxy:
    device = torch.device("cpu")

    def __init__(self, args, decode, seed, calls, session):
        self.args = args
        self.decode = decode
        self.seed = seed
        self.calls = calls
        self.session = session

    def generate(self, *, input_ids, attention_mask, max_new_tokens, eos_token_id, **options):
        prompt, = input_ids.tolist()
        assert bool(attention_mask.all())
        assert options["do_sample"] and max_new_tokens == self.decode["max_tokens"]
        assert options["temperature"] == self.decode["temperature"]
        assert options.get("top_p", self.decode["top_p"]) == self.decode["top_p"]
        assert options.get("top_k", self.decode["top_k"]) == self.decode["top_k"]
        assert [eos_token_id] == self.decode["stop_token_ids"]
        assert len(prompt) + max_new_tokens <= self.args.context
        seed = self.seed + 500 * len(self.calls)
        response = self.session.post(
            f"http://127.0.0.1:{self.args.port}/v1/completions",
            json={**self.decode, "model": self.args.model, "prompt": prompt,
                  "seed": seed, "return_token_ids": True},
            timeout=(3, 480),
        )
        response.raise_for_status()
        body = response.json()
        choice, = body["choices"]
        generated = choice["token_ids"]
        assert choice["prompt_token_ids"] == prompt
        assert len(generated) == body["usage"]["completion_tokens"] and 0 < len(generated) <= max_new_tokens
        return torch.tensor([prompt + generated], dtype=input_ids.dtype)


def instrument_client(client, calls):
    # Both backends use the released prompt templating, JSON extraction and text decoder.
    for method in ("generate", "generate_with_structured"):
        def record(answer, seconds, method=method):
            calls.append({"method": method, "seconds": seconds,
                          "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens})
        runner.record_call(client, method, record)
    return client


class SharedSelector(runner.PRMSelector):
    def __init__(self):
        super().__init__(str(runner.asset_path(runner.ASSETS[2])), device="cuda:0")
        runner.check_placement(self.model)
        self.lock = Lock()

    def score_reasoning(self, *args, **kwargs):
        queued = time.perf_counter()
        with self.lock:
            started = time.perf_counter()
            scores = super().score_reasoning(*args, **kwargs)
            seconds = time.perf_counter() - started
        THREAD.prm_queue += started - queued
        THREAD.prm_calls.append({**scores, "seconds": seconds})
        return scores


def dataset_for_run(args):
    dataset = runner.dataset(500)
    if args.sample:
        key = lambda row: hashlib.sha256(("published-sweep-20261124/" + row["dataset_unique_id"]).encode()).digest()
        dataset = sorted(dataset, key=key)[:args.sample]
    if args.shards > 1:
        dataset = [row for row in dataset if row["problem_id"] % args.shards == args.shard]
    return dataset


def solve(question, case, args, config, tokenizer, decode, native_model):
    index = question["problem_id"]
    destination = args.output / "problems" / f"{index:03d}"
    if destination.is_dir():
        return read(destination / "result.json")
    dynamic, iterations = CASES[case]
    failed = []
    attempts = 1 if args.strict_errors else 3
    for retry in range(attempts):
        attempt = Path(tempfile.mkdtemp(prefix=f"{index:03d}-", dir=args.output / "attempts"))
        seed = args.seed + index + 1_000_000 * retry
        calls, traces = [], []
        THREAD.prm_calls, THREAD.prm_queue = [], 0.0
        started = time.perf_counter()
        try:
            client = TransformerWrapper(runner.upstream.DictConfig(config["client"]))
            client._initialized = True
            client.tokenizer = tokenizer
            if native_model is None:
                if not hasattr(THREAD, "session"):
                    THREAD.session = requests.Session()
                    THREAD.session.trust_env = False
                client.model = GenerationProxy(args, decode, seed, calls, THREAD.session)
            else:
                torch.manual_seed(seed)
                client.model = native_model
            THREAD.client = instrument_client(client, calls)

            agent = runner.upstream.create_universal_agent(config)
            def record_trace(response, _seconds):
                traces.append({"structured_answer": response.answer,
                               "compute_strategy_outputs": response.compute_strategy_outputs,
                               "reasoner_outputs": response.reasoner_outputs})
            runner.record_call(agent, "solve", record_trace)
            core = runner.math_core.MATHCore(agent, runner.upstream.get_prompts())
            result, = runner.upstream.evaluate_reasoning_agent(
                agent_factory=lambda: core, dataset=[question], output_dir=str(attempt),
                agent_config=config["agent"], check_answer_fn=core.check_answer,
                extract_prediction_fn=core.extract_prediction, csv_filename_template="results_live.csv",
                json_filename_template="results_final.json", pathway_filename_template="pathway.json",
                prompt_kwargs={}, num_iterations=iterations, use_prm_selection=iterations > 1,
                prm_selection_metric="mean_reward", verbose=False, dataset_name="MATH",
                extra_problem_keys=["dataset_unique_id", "gold_answer_raw", "gold_answer_normalized", "solution"],
            )["results"]
            if not dynamic:
                assert [call["method"] for call in calls] == ["generate_with_structured", "generate"] * iterations
            result["observation"] = {
                "classification": CLASSIFICATION, "backend": args.backend, "seed": seed,
                "llm_calls": calls, "prm_calls": THREAD.prm_calls, "prm_queue_seconds": THREAD.prm_queue,
                "failed_attempts": failed, "traces": traces, "intra_iteration_scoring": [], "strategy_calls": [], "seconds": time.perf_counter() - started,
            }
            runner.write_json(attempt / "result.json", result)
            result = read(attempt / "result.json")
            check_problem(attempt, result, index, iterations, CLASSIFICATION)
            assert result["dataset_unique_id"] == question["dataset_unique_id"]
            attempt.rename(destination)
            return result
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, ValueError) as error:
            failure = {"type": type(error).__name__, "message": str(error), "seed": seed,
                       "llm_calls": len(calls), "seconds": time.perf_counter() - started}
            failed.append(failure)
            write(attempt / "failure.json", failure)
    raise RuntimeError(f"Problem {index} failed on {len(failed)} seed(s): {failed[-1]['type']}")


def run(args):
    repo = runner.ROOT / "upstream"
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == runner.UPSTREAM_COMMIT
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    canonical = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == DATASET_SHA256
    runtime = runner.require_h20()
    config = configuration(args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("problems", "attempts"):
        (args.output / name).mkdir(exist_ok=True)
    questions = dataset_for_run(args)
    if not questions:
        raise ValueError("No questions selected")
    started = time.perf_counter()
    native_model = None
    if args.backend == "transformers":
        native = TransformerWrapper(runner.upstream.DictConfig(config["client"]))
        native._initialize_client()
        runner.check_placement(native.model)
        raw_tokenizer, native_model = native.tokenizer, native.model
    else:
        raw_tokenizer = AutoTokenizer.from_pretrained(config["client"]["model_id"], local_files_only=True)
    decode = effective_decode(config, raw_tokenizer)
    if native_model is not None:
        assert native_model.generation_config.repetition_penalty == decode["repetition_penalty"]
        assert native_model.generation_config.top_k == decode["top_k"]
    tokenizer = LockedTokenizer(raw_tokenizer)
    setup = {
        "case": args.case, "backend": args.backend, "concurrency": args.workers,
        "runtime": runtime, "package_version": args.vllm_version,
        "upstream_commit": runner.UPSTREAM_COMMIT, "dataset_sha256": DATASET_SHA256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decode": decode, "seed": args.seed, "strict_errors": args.strict_errors,
        "sample_ids": [row["problem_id"] for row in questions], "endpoint_port": args.port,
        "agent": config["agent"], "client": config["client"], "eval": config["eval"],
        "datasets_and_models": [list(asset) for asset in runner.ASSETS],
    }
    config_path = args.output / "config.json"
    if config_path.exists():
        if read(config_path) != json.loads(json.dumps(setup)):
            raise ValueError("Resume requires the same configuration")
    else:
        write(config_path, setup)

    if (args.output / "summary.json").exists():
        previous = read(args.output / "summary.json")
        if previous["samples"] == len(questions) and not previous["failed"]:
            print("METRICS " + json.dumps(previous), flush=True)
            return True

    # Limit output from the upstream evaluator without changing the evaluator itself.
    settled, failures = {}, {}
    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as detail:
        with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail), contextlib.ExitStack() as stack:
            prm = SharedSelector() if CASES[args.case][1] > 1 else None
            if prm is not None:
                stack.enter_context(patch.object(runner.upstream, "PRMSelector", lambda: prm))
            stack.enter_context(patch.object(runner.upstream, "create_llm_client", lambda _config: THREAD.client))
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
            initialized = time.perf_counter()
            futures = {pool.submit(solve, row, args.case, args, config, tokenizer, decode, native_model): row
                       for row in questions}
            for future in as_completed(futures):
                question = futures[future]
                index = question["problem_id"]
                try:
                    row = future.result()
                    assert row["dataset_unique_id"] == question["dataset_unique_id"]
                    settled[index] = row
                except Exception as error:
                    logging.exception("Uncommitted problem %s", index)
                    failures[index] = {"type": type(error).__name__, "message": str(error)}
                done = len(settled) + len(failures)
                if done == len(questions) or done % 8 == 0:
                    print(f"{args.case} c{args.workers}: {len(settled)}/{len(questions)} complete, {len(failures)} failed",
                          file=sys.__stdout__, flush=True)
            elapsed = time.perf_counter() - initialized

    ordered = [settled[index] for index in sorted(settled)]
    manifest = [{"problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"],
                 "predicted_answer": row["final_predicted_answer"], "is_correct": row["final_is_correct"],
                 "selected_iteration": row["selected_iteration_idx"], "seed": row["observation"]["seed"],
                 "llm_calls": len(row["observation"]["llm_calls"]),
                 "output_tokens": sum(call["output_tokens"] for call in row["observation"]["llm_calls"])}
                for row in ordered]
    write(args.output / "predictions.json", manifest)
    observes = [row["observation"] for row in ordered]
    calls = [call for obs in observes for call in obs["llm_calls"]]
    scores = [score for obs in observes for score in obs["prm_calls"]]
    correct = sum(row["is_correct"] is True for row in manifest)
    tokens = sum(call["output_tokens"] for call in calls)
    summary = {
        "case": args.case, "classification": CLASSIFICATION, "backend": args.backend,
        "workers": args.workers, "samples": len(manifest), "requested": len(questions), "correct": correct,
        "accuracy": correct / len(manifest) if manifest else None, "failed": failures,
        "answer_verdicts": dict(Counter(str(row["is_correct"]) for row in manifest)),
        "seconds": elapsed, "setup_seconds": initialized - started,
        "questions_per_minute": 60 * len(manifest) / elapsed, "output_tokens_per_second": tokens / elapsed,
        "output_tokens": tokens, "generation_calls": len(calls), "prm_calls": len(scores),
        "prm_compute_seconds": sum(score["seconds"] for score in scores),
        "prm_queue_seconds": sum(obs["prm_queue_seconds"] for obs in observes),
        "question_retries": sum(len(obs["failed_attempts"]) for obs in observes),
        "prm_all_zero_questions": sum(bool(obs["prm_calls"]) and all(s["mean_reward"] == 0 for s in obs["prm_calls"])
                                      for obs in observes),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "native_files_verified_at_commit": True, "pinned_dataset_checked": True,
    }
    write(args.output / "summary.json", summary)
    print("METRICS " + json.dumps(summary), flush=True)
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="vllm")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=18420)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--model", default="published-qwen25")
    parser.add_argument("--vllm-version")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--strict-errors", action="store_true")
    args = parser.parse_args()
    assert args.workers >= 1 and (args.sample is None or 1 <= args.sample <= 500)
    assert 0 <= args.shard < args.shards
    assert args.backend != "transformers" or args.workers == 1
    assert args.backend != "vllm" or args.vllm_version
    raise SystemExit(0 if run(args) else 1)

"""Benchmark or run the pinned public MATH-500 code with native or batched decoding."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
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


def load_benchmark(args):
    if args.dataset == "math":
        canonical = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == DATASET_SHA256
        return runner.dataset(500), {"split": "test", "sha256": DATASET_SHA256}
    if args.dataset_manifest is None:
        raise ValueError("--dataset-manifest is required for AIME24 and AMO-Bench")

    from datasets import Dataset

    spec = read(args.dataset_manifest)[args.dataset]
    expected = {
        "aime24": ("Maxwell-Jia/AIME_2024", "train", 30, "ID", runner.math_core.AIME24Core),
        "amo": ("meituan-longcat/AMO-Bench", "test", 50, "question_id", runner.math_core.AMOCore),
    }[args.dataset]
    repo, split, count, id_field, core_class = expected
    assert (spec["repo"], spec["split"], spec["samples"]) == (repo, split, count)
    parquet = args.dataset_manifest.parent / f"{args.dataset}.parquet"
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == spec["sha256"]
    source = Dataset.from_parquet(str(parquet))
    assert len(source) == count

    def pinned_dataset(name):
        assert name == repo
        return {split: source}

    with patch.object(runner.math_core, "load_dataset", pinned_dataset):
        questions = core_class.load_dataset(split=split)
    for index, row in enumerate(questions):
        row["dataset_unique_id"] = str(source[index][id_field])
    assert len(questions) == count == len({row["dataset_unique_id"] for row in questions})
    assert all(row["problem"] and row["gold_answer_extracted"] for row in questions)
    return questions, spec


class ResidentSelector(runner.PRMSelector):
    """Use the same upstream selector with a loaded local PRM scoring process."""

    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}/score"

    def score_reasoning(self, problem, reasoning_steps, system_prompt):
        started = time.perf_counter()
        response = THREAD.session.post(self.url, json={
            "problem": problem, "reasoning_steps": reasoning_steps, "system_prompt": system_prompt,
        }, timeout=(3, 240))
        if response.status_code == 422 and response.json().get("error") == "The scorer returned no step rewards":
            # The released selector records a zero score when scoring yields no steps.
            scores = {"step_rewards": [], "mean_reward": 0., "min_reward": 0., "final_reward": 0., "num_steps": 0}
        else:
            response.raise_for_status()
            scores = response.json()
            assert scores.pop("model") == "Qwen2.5-Math-PRM-7B"
        THREAD.prm_calls.append({**scores, "seconds": time.perf_counter() - started})
        return scores


def dataset_for_run(args, dataset):
    if args.sample:
        key = lambda row: hashlib.sha256(("published-sweep-20261124/" + (args.dataset + "/" if args.dataset != "math" else "") + row["dataset_unique_id"]).encode()).digest()
        dataset = sorted(dataset, key=key)[:args.sample]
    if args.shards > 1:
        dataset = [row for row in dataset if row["problem_id"] % args.shards == args.shard]
    if args.priority_file:
        ranked = read(args.priority_file)["cases"][args.case]["priority_question_ids"]
        positions = {index: position for position, index in enumerate(ranked)}
        dataset.sort(key=lambda row: (positions.get(row["problem_id"], len(positions)), row["problem_id"]))
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
            core_class = {"math": runner.math_core.MATHCore, "aime24": runner.math_core.AIME24Core,
                          "amo": runner.math_core.AMOCore}[args.dataset]
            core = core_class(agent, runner.upstream.get_prompts())
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


def shared_completions(pool, prepare, questions, claim, workers):
    by_id = {row["problem_id"]: row for row in questions}
    active = {}

    def submit_one():
        question_id = claim()
        if question_id is None:
            return False
        row = by_id[question_id]
        active[pool.submit(prepare, row)] = row
        return True

    for _ in range(workers):
        if not submit_one():
            break
    while active:
        finished, _ = wait(active, return_when=FIRST_COMPLETED)
        for future in finished:
            yield future, active.pop(future)
            submit_one()


def run(args, *, claim_question=None):
    repo = runner.ROOT / "upstream"
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == runner.UPSTREAM_COMMIT
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])
    dataset, dataset_spec = load_benchmark(args)
    runtime = runner.require_h20()
    config = configuration(args.case)
    config["eval"].update(max_problems=len(dataset), split=dataset_spec["split"])
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("problems", "attempts"):
        (args.output / name).mkdir(exist_ok=True)
    questions = dataset_for_run(args, dataset)
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
        "upstream_commit": runner.UPSTREAM_COMMIT, "dataset_sha256": dataset_spec["sha256"],
        "dataset": args.dataset, "dataset_spec": dataset_spec, "prm_score_port": args.prm_port,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decode": decode, "seed": args.seed, "strict_errors": args.strict_errors,
        "sample_ids": sorted(row["problem_id"] for row in questions) if args.shards > 1
                      else [row["problem_id"] for row in questions],
        "submission_order": [row["problem_id"] for row in questions],
        "endpoint_port": args.port,
        "agent": config["agent"], "client": config["client"], "eval": config["eval"],
        "datasets_and_models": [list(asset) for asset in runner.ASSETS[1:]],
    }
    if claim_question is not None:
        setup["assignment"] = "shared_head_tail"
    config_path = args.output / "config.json"
    if config_path.exists():
        if read(config_path) != json.loads(json.dumps(setup)):
            raise ValueError("Resume requires the same configuration")
    else:
        write(config_path, setup)

    if claim_question is None and (args.output / "summary.json").exists():
        previous = read(args.output / "summary.json")
        if previous["samples"] == len(questions) and not previous["failed"]:
            print("METRICS " + json.dumps(previous), flush=True)
            return True

    # Limit output from the upstream evaluator without changing the evaluator itself.
    settled, failures = {}, {}
    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as detail:
        with contextlib.redirect_stdout(detail), contextlib.redirect_stderr(detail), contextlib.ExitStack() as stack:
            if CASES[args.case][1] > 1:
                prm = ResidentSelector(args.prm_port) if args.prm_port else SharedSelector()
                stack.enter_context(patch.object(runner.upstream, "PRMSelector", lambda: prm))
            stack.enter_context(patch.object(runner.upstream, "create_llm_client", lambda _config: THREAD.client))
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
            initialized = time.perf_counter()
            def evaluate(row):
                return solve(row, args.case, args, config, tokenizer, decode, native_model)

            if claim_question is None:
                futures = {pool.submit(evaluate, row): row for row in questions}
                completions = ((future, futures[future]) for future in as_completed(futures))
            else:
                completions = shared_completions(pool, evaluate, questions, claim_question, args.workers)
            for future, question in completions:
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
                    total = "shared" if claim_question is not None else len(questions)
                    print(f"{args.case} c{args.workers}: {len(settled)}/{total} complete, {len(failures)} failed",
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
        "dataset": args.dataset, "prm_timing_includes_http": bool(args.prm_port and CASES[args.case][1] > 1),
        "workers": args.workers, "samples": len(manifest),
        "requested": len(questions) if claim_question is None else len(manifest) + len(failures), "correct": correct,
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
    parser.add_argument("--dataset", choices=("math", "aime24", "amo"), default="math")
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--prm-port", type=int)
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
    parser.add_argument("--priority-file", type=Path)
    parser.add_argument("--strict-errors", action="store_true")
    args = parser.parse_args()
    assert args.workers >= 1 and (args.sample is None or 1 <= args.sample <= 500)
    assert 0 <= args.shard < args.shards
    assert args.backend != "transformers" or args.workers == 1
    assert args.backend != "vllm" or args.vllm_version
    assert args.prm_port is None or args.backend == "vllm"
    raise SystemExit(0 if run(args) else 1)

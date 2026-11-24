"""Run MATH-500 with published-code smoke cases or the paper-described protocol."""
import argparse
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import random
import subprocess
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
import yaml
from datasets import load_dataset

from validate_models import ASSETS, ROOT, PRMSelector, asset_path, check_placement, require_h20

import main as upstream
from src import math_core, universal_agent
from paper_protocol import PaperAgent, PaperPRM, PaperSearch

CASES = {"A": (10, 1, False), "B": (10, 10, False), "C": (20, 3, True)}
PAPER_CASES = {"E1": (500, 1, False), "E2": (500, 10, False),
               "E3": (500, 1, True), "E4": (500, 10, True)}
UPSTREAM_COMMIT = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"


def configure(case, seed, limit=None):
    paper = case in PAPER_CASES
    count, iterations, dynamic = (PAPER_CASES if paper else CASES)[case]
    if limit is not None:
        if not paper or not 1 <= limit <= count:
            raise ValueError("--limit must be between 1 and 500 and is supported for E1–E4")
        count = limit
    config = upstream.load_config(str(ROOT / "upstream" / "config.yaml"))
    config["agent"].update(mode="dynamic" if dynamic else "direct",
                           use_planner=dynamic, use_tool_selector=dynamic, use_compute_selector=dynamic)
    config["eval"].update(max_problems=count, num_iterations=iterations, use_prm_selection=iterations > 1,
                          output_dir=None, num_workers=1)
    config["client"].update(client_name="transformer", base_url=None, model_id=str(asset_path(ASSETS[1])),
                            torch_dtype="bfloat16", device_map="cuda:0", trust_remote_code=False)
    config["reproduction"] = {
        "classification": "smoke_published_code",
        "case": case,
        "seed": seed,
        "question_seed_rule": "base seed + MATH-500 test row index",
        "upstream_commit": UPSTREAM_COMMIT,
        "datasets_and_models": ASSETS,
    }
    if paper:
        config["agent"].update(fixed_tool=None, fixed_compute=None)
        config["client"]["generate_kwargs"].update(temperature=0.7, top_p=0.9, max_tokens=1024)
        config["reproduction"].update(
            classification="paper_description" if count == 500 else "smoke_paper_description",
            label="按论文描述复现" if count == 500 else "按论文描述复现（小规模验证）",
            source_sha256={
                name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                for name in ("run_smoke.py", "paper_protocol.py")
            },
        )
    return config


def dataset(count):
    _, repo, _ = ASSETS[0]
    path = asset_path(ASSETS[0]) / "test.jsonl"
    splits = load_dataset("json", data_files={"test": str(path)})
    assert len(splits["test"]) == 500

    def pinned_load(name):
        assert name == repo
        return splits

    with patch.object(math_core, "load_dataset", pinned_load):
        questions = math_core.MATHCore.load_dataset(split="test", max_problems=count)
    for index, question in enumerate(questions):
        question["dataset_unique_id"] = splits["test"][index]["unique_id"]
    return questions


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        print(json.dumps(value, indent=2, ensure_ascii=False, default=lambda obj: obj.item()), file=output)


def record_call(owner, name, observe):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        output = original(*args, **kwargs)
        observe(output, time.perf_counter() - start)
        return output

    setattr(owner, name, wrapped)


def evaluate_question(agent, prm, config, question, attempt, strategy_factory=None):
    core = math_core.MATHCore(agent, upstream.get_prompts())
    prm_factory = (lambda: prm) if prm is not None else PRMSelector
    strategy_factory = strategy_factory or universal_agent.get_compute_strategy
    with patch.object(upstream, "PRMSelector", prm_factory), patch.object(
        universal_agent, "get_compute_strategy", strategy_factory
    ):
        return upstream.evaluate_reasoning_agent(
            agent_factory=lambda: core, dataset=[question], output_dir=str(attempt),
            agent_config=config["agent"], check_answer_fn=core.check_answer,
            extract_prediction_fn=core.extract_prediction, csv_filename_template="results_live.csv",
            json_filename_template="results_final.json", pathway_filename_template="pathway.json",
            prompt_kwargs={}, num_iterations=config["eval"]["num_iterations"],
            use_prm_selection=config["eval"]["use_prm_selection"], prm_selection_metric="mean_reward",
            extra_problem_keys=["dataset_unique_id", "gold_answer_raw", "gold_answer_normalized", "solution"],
            verbose=False, dataset_name="MATH",
        )["results"][0]


def run(config, output, resume):
    count = config["eval"]["max_problems"]
    paper = config["reproduction"]["case"] in PAPER_CASES
    search_with_prm = paper and config["agent"]["use_compute_selector"]
    final_selection = config["eval"]["use_prm_selection"]
    assert subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip() == UPSTREAM_COMMIT
    assert not subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "status", "--porcelain", "--untracked-files=no"], text=True)
    config["eval"]["output_dir"] = str(output)
    config["reproduction"]["runtime"] = require_h20()
    config["reproduction"]["dataset_sha256"] = hashlib.sha256((asset_path(ASSETS[0]) / "test.jsonl").read_bytes()).hexdigest()

    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settled = output / "problems"
        settled.mkdir(exist_ok=True)
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        config_file = output / "config.yaml"
        config_text = yaml.safe_dump(config, sort_keys=False)
        if config_file.exists():
            assert resume and config_file.read_text() == config_text, "Resume requires the same configuration and runtime"
        else:
            with config_file.open("x") as stream:
                stream.write(config_text)
        if (output / "summary.json").exists():
            print((output / "summary.json").read_text())
            return

        questions = dataset(count)
        start = time.perf_counter()
        if paper:
            with patch.object(upstream, "UniversalAgent", PaperAgent):
                agent = upstream.create_universal_agent(config)
        else:
            agent = upstream.create_universal_agent(config)
        agent.client._initialize_client()
        check_placement(agent.client.model)
        prm = None
        if final_selection or search_with_prm:
            prm_type = PaperPRM if paper else PRMSelector
            prm = prm_type(str(asset_path(ASSETS[2])), device="cuda:0")
        if prm is not None:
            check_placement(prm.model)
        model_load_seconds = time.perf_counter() - start
        print(f"Models ready after {model_load_seconds:.3f} seconds", flush=True)

        generation_calls, prm_calls, intra_calls, traces, strategy_calls = [], [], [], [], []
        for method in ("generate", "generate_with_structured"):
            def observe_generation(response, seconds, method=method):
                generation_calls.append({"method": method, "seconds": seconds,
                                         "input_tokens": response.input_tokens, "output_tokens": response.output_tokens})
            record_call(agent.client, method, observe_generation)

        def observe_trace(response, seconds):
            traces.append({"structured_answer": response.answer,
                           "compute_strategy_outputs": response.compute_strategy_outputs,
                           "reasoner_outputs": response.reasoner_outputs})
        record_call(agent, "solve", observe_trace)
        if agent.prm_model is not None and not paper:
            record_call(agent.prm_model, "score_response",
                        lambda score, seconds: intra_calls.append({"score": score, "seconds": seconds}))
        if search_with_prm:
            record_call(prm, "score_search",
                        lambda scores, seconds: intra_calls.append({"seconds": seconds, **scores}))
        if final_selection:
            record_call(prm, "score_reasoning",
                        lambda scores, seconds: prm_calls.append({"seconds": seconds, **scores}))

        upstream_strategy_factory = universal_agent.get_compute_strategy

        def observed_strategy(strategy_name, prm_model, rng=None):
            if search_with_prm:
                strategy = PaperSearch(strategy_name, prm, rng)
            else:
                strategy = upstream_strategy_factory(strategy_name, prm_model, rng)
            llm_start = len(generation_calls)
            scorer_start = len(intra_calls)

            def observe_execution(output, seconds):
                calls = generation_calls[llm_start:]
                strategy_calls.append({
                    "strategy": strategy_name, "seconds": seconds,
                    "llm_calls": len(calls), "local_scoring_calls": len(intra_calls) - scorer_start,
                    "input_tokens": sum(call["input_tokens"] for call in calls),
                    "output_tokens": sum(call["output_tokens"] for call in calls),
                })

            record_call(strategy, "execute", observe_execution)
            return strategy

        for question in questions:
            index = question["problem_id"]
            destination = settled / f"{index:03d}"
            if destination.is_dir():
                assert (destination / "result.json").is_file()
                continue
            seed = config["reproduction"]["seed"] + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            for records in (generation_calls, prm_calls, intra_calls, traces, strategy_calls):
                records.clear()
            attempt = Path(tempfile.mkdtemp(prefix=f"{index:03d}-", dir=attempts))
            question_start = time.perf_counter()
            try:
                result = evaluate_question(agent, prm, config, question, attempt, observed_strategy)
                if final_selection:
                    assert len(prm_calls) == config["eval"]["num_iterations"] and all(
                        paper or entry["num_steps"] > 0 for entry in prm_calls
                    ), "Unexpected final PRM scoring records"
                result["observation"] = {
                    "seed": seed, "seconds": time.perf_counter() - question_start,
                    "llm_calls": generation_calls, "prm_calls": prm_calls, "intra_iteration_scoring": intra_calls,
                    "traces": traces, "strategy_calls": strategy_calls, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                if paper:
                    result["observation"]["classification"] = config["reproduction"]["classification"]
                write_json(attempt / "result.json", result)
                attempt.rename(destination)
            except Exception as error:
                write_json(attempt / "failure.json", {"type": type(error).__name__, "message": str(error)})
                raise
            print(f"Completed {index + 1}/{count} in {result['observation']['seconds']:.2f}s", flush=True)

        results = [json.loads((settled / f"{i:03d}" / "result.json").read_text()) for i in range(count)]
        observations = [r["observation"] for r in results]
        correct = sum(bool(r["final_is_correct"]) for r in results)
        seconds = sum(r["seconds"] for r in observations)
        calls = [call for r in observations for call in r["llm_calls"]]
        scores = [call for r in observations for call in r["prm_calls"]]
        intra_scores = [call for r in observations for call in r["intra_iteration_scoring"]]
        summary = {
            "case": config["reproduction"]["case"], "samples": count, "correct": correct,
            "accuracy": correct / count, "model_load_seconds_last_attempt": model_load_seconds,
            "problem_wall_seconds": seconds, "mean_seconds_per_question": seconds / count,
            "gpu_peak_allocated_bytes": max(r["gpu_peak_allocated_bytes"] for r in observations),
            "gpu_peak_reserved_bytes": max(r["gpu_peak_reserved_bytes"] for r in observations),
            "llm_calls": len(calls), "llm_seconds": sum(c["seconds"] for c in calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "prm_calls": len(scores), "prm_seconds": sum(c["seconds"] for c in scores),
            "intra_iteration_scoring_calls": len(intra_scores),
            "intra_iteration_scoring_seconds": sum(c["seconds"] for c in intra_scores),
        }
        if paper:
            summary.update(classification=config["reproduction"]["classification"],
                           label=config["reproduction"]["label"])
        with tempfile.TemporaryDirectory(dir=output) as temp:
            pending = Path(temp) / "summary.json"
            write_json(pending, summary)
            pending.rename(output / "summary.json")
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[*CASES, *PAPER_CASES])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Run the first N MATH-500 problems for a paper-protocol check")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 500:
        parser.error("--seed must be between 0 and 2**32 - 500")
    try:
        config = configure(args.case, args.seed, args.limit)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO)
    run(config, args.output.resolve(), args.resume)


if __name__ == "__main__":
    main()

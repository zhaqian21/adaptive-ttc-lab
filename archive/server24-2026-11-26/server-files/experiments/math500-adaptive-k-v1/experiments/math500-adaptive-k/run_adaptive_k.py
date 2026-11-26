"""Run the four paired MATH-500 policies or the standalone early-stop policy."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import revision_policy
import run_math500 as revision
import sampling
from sampling import POLICIES, TOKEN_LIMIT, Prefix, collect, costs, select


baseline = revision.baseline
runner = revision.runner


def solve(question, args, config, tokenizer, decode):
    index = question["problem_id"]
    path = args.output / "problems" / f"{index:03d}.json"
    if path.exists():
        if baseline.read(path)["dataset_unique_id"] != question["dataset_unique_id"]:
            raise ValueError(f"Wrong dataset row at {index}")
        return

    problem = question["problem"]
    evaluator = runner.upstream.get_evaluator("MATH")
    with baseline.requests.Session() as session:
        session.trust_env = False
        sample, _ = revision.make_sampler(problem, index, evaluator, args, config, tokenizer, decode, session)
        roots, attempts, budget, prefixes = collect(sample, args.mode)
        count = max((p.candidates for p in prefixes.values() if p.reason != "agreement"), default=0)
        scorer = revision.ResidentSelector(args.prm_port, session)
        scores = []
        if count:
            selected, _, scores = scorer.select_best_iteration(
                problem, [root.scoring_entry for root in roots[:count]], "mean_reward", revision.SYSTEM_PROMPT,
            )
            if (len(scores) != count or len(scorer.calls) != count
                    or selected != select(scores, Prefix(count, 0, "prm"))):
                raise ValueError("Unexpected upstream PRM selection or accounting")

    candidates = [
        {"prediction": root.prediction,
         "correct": bool(evaluator._check_answers_equiv(root.prediction, question["gold_answer_extracted"])),
         "mean_reward": scores[i]["mean_reward"] if i < count else None,
         "prm_seconds": scorer.calls[i]["seconds"] if i < count else None}
        for i, root in enumerate(roots)
    ]
    arms = {}
    for name, prefix in prefixes.items():
        selected = select(candidates, prefix)
        chosen = candidates[selected] if selected is not None else None
        calls = 0 if prefix.reason == "agreement" else prefix.candidates
        arms[name] = {
            **asdict(prefix),
            **costs(budget.calls, prefix.calls),
            "prm_calls": calls,
            "prm_request_seconds": sum(c["prm_seconds"] for c in candidates[:calls]),
            "selected_index": selected,
            "prediction": chosen["prediction"] if chosen else None,
            "correct": chosen["correct"] if chosen else False,
        }
    baseline.write(path, {
        "problem_id": index, "dataset_unique_id": question["dataset_unique_id"], "attempts": attempts,
        "calls": budget.calls, "budget_boundary": budget.boundary, "candidates": candidates, "policies": arms,
    })


def run(args):
    repo = runner.ROOT / "upstream"
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if sha != runner.UPSTREAM_COMMIT or subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
    ):
        raise ValueError("Expected the pinned upstream checkout without tracked changes")
    dataset = runner.asset_path(runner.ASSETS[0]) / "test.jsonl"
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != baseline.DATASET_SHA256:
        raise ValueError("Unexpected MATH-500 data digest")

    args.case, args.priority_file, args.shards = "E4", None, 2
    config = baseline.configuration(args.case)
    raw_tokenizer = baseline.AutoTokenizer.from_pretrained(config["client"]["model_id"], local_files_only=True)
    decode = baseline.effective_decode(config, raw_tokenizer)
    tokenizer = baseline.LockedTokenizer(raw_tokenizer)
    questions = sorted(baseline.dataset_for_run(args), key=lambda row: row["problem_id"])
    if not questions:
        raise ValueError("Empty shard")
    sources = (__file__, revision.__file__, revision_policy.__file__, sampling.__file__, baseline.__file__)
    setup = {
        "mode": args.mode, "upstream_commit": sha,
        "source_sha256": {Path(file).name: hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in sources},
        "dataset_sha256": baseline.DATASET_SHA256, "assets": [list(asset) for asset in runner.ASSETS],
        "vllm_version": args.vllm_version, "model": args.model, "sample_decode": decode, "seed": args.seed,
        "policies": list(POLICIES) if args.mode == "paired" else ["adaptive"],
        "token_limit": TOKEN_LIMIT, "fixed_counts": sampling.FIXED_COUNTS, "agreement_count": 3,
        "max_attempts": sampling.MAX_ATTEMPTS, "prm_metric": "mean_reward",
        "workers": args.workers, "shards": args.shards, "shard": args.shard, "sample": args.sample,
        "question_ids": [row["problem_id"] for row in questions],
    }
    (args.output / "problems").mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and baseline.read(config_path) != setup:
        raise ValueError("Checkpoint configuration differs from this run")
    baseline.write(config_path, setup)

    started = time.perf_counter()
    failures, completed = {}, 0
    with (args.output / "details.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(log), patch.object(runner.upstream, "create_llm_client", lambda _: baseline.THREAD.client):
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(solve, q, args, config, tokenizer, decode): q["problem_id"] for q in questions}
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                    except Exception as error:
                        traceback.print_exc(file=log)
                        failures[futures[future]] = f"{type(error).__name__}: {error}"
                    if (completed + len(failures)) % 8 == 0 or completed + len(failures) == len(questions):
                        print(f"shard {args.shard}: {completed}/{len(questions)} complete, {len(failures)} failed",
                              file=sys.__stdout__, flush=True)
    baseline.write(args.output / "execution.json", {
        "seconds": time.perf_counter() - started, "complete": completed,
        "requested": len(questions), "failures": failures,
    })
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("paired", "adaptive"), default="paired")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prm-port", type=int, default=18422)
    parser.add_argument("--shard", type=int, required=True, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--sample", type=int, choices=range(2, 501))
    parser.add_argument("--model", default="published-qwen25")
    parser.add_argument("--vllm-version", default="0.33.0")
    args = parser.parse_args()
    args.seed = 0
    if args.workers < 1:
        parser.error("Concurrency must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

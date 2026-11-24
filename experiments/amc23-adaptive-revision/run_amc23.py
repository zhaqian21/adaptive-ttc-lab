"""Run and audit the three adaptive revision policies on the pinned AMC23 split."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_math500 import load, shard_files, summarize_policies, validate_result
from revision_policy import POLICIES


DATASET = {
    "repo": "math-ai/amc23",
    "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
    "split": "test",
    "file": "test-00000-of-00001.parquet",
    "sha256": "b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e",
    "samples": 40,
}


def load_amc23(path):
    from datasets import Dataset
    from run_math500 import runner

    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET["sha256"]:
        raise ValueError("AMC23 parquet does not match the pinned digest")
    rows = Dataset.from_parquet(str(path))
    if len(rows) != DATASET["samples"] or len(set(rows["id"])) != len(rows):
        raise ValueError("Expected 40 unique AMC23 problems")
    evaluator = runner.upstream.get_evaluator("MATH")
    questions = []
    for index, row in enumerate(rows):
        answer = str(row["answer"]).strip()
        gold = evaluator._extract_answer_from_model_completion(r"\boxed{" + answer + "}")
        if not row["question"].strip() or not gold or not evaluator._check_answers_equiv(gold, answer):
            raise ValueError(f"AMC23 grading control failed at source ID {row['id']}")
        questions.append({"problem_id": index, "dataset_unique_id": str(row["id"]),
                          "problem": row["question"], "gold_answer_extracted": gold})
    return questions


def for_shard(questions, lane, sample_per_shard):
    rows = questions[lane::2]
    if sample_per_shard is not None:
        def sample_key(row):
            return hashlib.sha256(f"amc23-d-v1/{row['dataset_unique_id']}".encode()).digest()

        rows = sorted(rows, key=sample_key)[:sample_per_shard]
    return sorted(rows, key=lambda row: row["problem_id"])


def benchmark(args):
    manifest = {
        "dataset": DATASET,
        "dataset_sha256": DATASET["sha256"],
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sample_per_shard": args.sample_per_shard,
    }
    return for_shard(load_amc23(args.dataset_file), args.shard, args.sample_per_shard), manifest


def audit(root, dataset_file, sample_per_shard):
    from run_math500 import runner

    questions = load_amc23(dataset_file)
    evaluator = runner.upstream.get_evaluator("MATH")
    configs, executions, predictions = [], [], []
    manifest_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for lane in (0, 1):
        directory = root / f"shard-{lane}"
        config, execution, files = shard_files(directory, lane)
        expected = for_shard(questions, lane, sample_per_shard)
        if (config["dataset"] != DATASET or config["dataset_sha256"] != DATASET["sha256"]
                or config["benchmark_source_sha256"] != manifest_hash
                or config["sample_per_shard"] != sample_per_shard or execution["failures"]
                or config["question_ids"] != [row["problem_id"] for row in expected]):
            raise ValueError(f"Incomplete AMC23 shard {lane}")
        configs.append(config)
        executions.append(execution)
        for question in expected:
            index = question["problem_id"]
            row = load(files[index])
            validate_result(row, config, index, question["dataset_unique_id"])
            for policy in row["policies"].values():
                candidates = policy["candidates"]
                actual = [bool(evaluator._check_answers_equiv(c["prediction"],
                                                           question["gold_answer_extracted"])) for c in candidates]
                if actual != [c["correct"] for c in candidates]:
                    raise ValueError(f"Inconsistent candidate labels in AMC23 row {index}")
                policy["has_correct_candidate"] = any(actual)
                policy["prm_missed_correct_candidate"] = any(actual) and not policy["correct"]
            predictions.append(row)

    comparable = [{key: value for key, value in config.items() if key not in ("shard", "question_ids")}
                  for config in configs]
    if comparable[0] != comparable[1]:
        raise ValueError("AMC23 shard configurations differ")
    predictions.sort(key=lambda row: row["problem_id"])
    summaries = summarize_policies(predictions)
    for policy in POLICIES:
        runs = [row["policies"][policy] for row in predictions]
        coverage = sum(run["has_correct_candidate"] for run in runs)
        missed = sum(run["prm_missed_correct_candidate"] for run in runs)
        if coverage != summaries[policy]["correct"] + missed:
            raise ValueError(f"Candidate coverage does not reconcile for {policy}")
        summaries[policy].update(candidate_pool_correct=coverage, prm_missed_correct=missed)
    version = root / "export"
    version.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in predictions).encode("utf-8")
    (version / "predictions.jsonl").write_bytes(payload)
    summary = {
        "dataset": DATASET, "samples": len(predictions), "formal": sample_per_shard is None,
        "wall_seconds": max(execution["seconds"] for execution in executions),
        "shard_counts": [execution["complete"] for execution in executions],
        "policies": summaries, "configs": configs,
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (version / "summary.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({key: value for key, value in summary.items() if key != "configs"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-shard", type=int, choices=range(1, 21))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--shard", type=int, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--vllm-version", default="0.33.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prm-port", type=int, default=18422)
    args = parser.parse_args()
    if args.audit:
        audit(args.output, args.dataset_file, args.sample_per_shard)
        return 0
    if args.shard is None or args.port is None or args.workers < 1:
        parser.error("A worker requires its shard, vLLM port, and positive concurrency")
    from run_math500 import run

    args.context, args.model = 32768, "published-qwen25"
    args.seed, args.token_budget, args.initial_roots = 0, 125000, 3
    args.shards, args.only_row = 2, None
    return run(args, load_benchmark=benchmark)


if __name__ == "__main__":
    raise SystemExit(main())

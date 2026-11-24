"""Audit the module decisions and merge the two formal shards."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from statistics import median

from audit_math500 import load, shard_files
from replay_modules import labelled, replay
from run_modules import base
from routing import METHODS
from sampling import TOKEN_LIMIT


def audit(root, wall_seconds):
    configs, executions, rows, retries = [], [], [], []
    evaluator = base.runner.upstream.get_evaluator("MATH")
    for shard in (0, 1):
        directory = root / f"shard-{shard}"
        config, execution, files = shard_files(directory, shard)
        method = config["extension"]["method"]
        if method not in METHODS or execution["failures"] or config["dataset_sha256"] != base.baseline.DATASET_SHA256:
            raise ValueError(f"Unexpected setup or incomplete shard {shard}")
        args = SimpleNamespace(shard=shard, shards=2, sample=config["sample"], case="E4", priority_file=None)
        questions = {q["problem_id"]: q for q in base.baseline.dataset_for_run(args)}
        if set(questions) != set(files):
            raise ValueError(f"Wrong coverage on shard {shard}")

        for index, filename in sorted(files.items()):
            row = load(filename)
            q = questions[index]
            if row["problem_id"] != index or row["dataset_unique_id"] != q["dataset_unique_id"]:
                raise ValueError(f"Wrong row at {index}")
            roots, attempts, budget, rewards, checks, arms = replay(row, method, TOKEN_LIMIT)
            if (len(roots) != len(row["candidates"]) or attempts != row["attempts"] or budget.calls != row["calls"]
                    or budget.boundary != row["budget_boundary"] or list(rewards) != row["prm_indices"]
                    or checks != row["route_checks"] or labelled(arms, row["candidates"]) != row["policies"]):
                raise ValueError(f"Trace or policy mismatch at {index}")
            for i, candidate in enumerate(row["candidates"]):
                grade = bool(evaluator._check_answers_equiv(candidate["prediction"], q["gold_answer_extracted"]))
                if candidate["correct"] != grade or candidate["mean_reward"] != rewards.get(i):
                    raise ValueError(f"Incorrect score or label at {index}/{i}")
            if any(not math.isfinite(row[k]) or row[k] < 0 for k in ("inference_seconds", "prm_total_seconds")):
                raise ValueError(f"Invalid time at {index}")
            rows.append(row)
        configs.append(config)
        executions.append(execution)
        if (directory / "first-attempt.json").exists():
            failure = load(directory / "first-attempt.json")
            retries.append({"shard": shard, "failed_rows": sorted(map(int, failure["failures"])),
                            "first_attempt_seconds": failure["seconds"]})

    if {k: v for k, v in configs[0].items() if k not in ("shard", "question_ids")} != {
        k: v for k, v in configs[1].items() if k not in ("shard", "question_ids")
    }:
        raise ValueError("The two shards share different setups")
    rows.sort(key=lambda row: row["problem_id"])
    sample = configs[0]["sample"]
    if (len(rows) != (sample or 500) or len({row["dataset_unique_id"] for row in rows}) != len(rows)
            or (sample is None and [row["problem_id"] for row in rows] != list(range(500)))):
        raise ValueError("Incomplete benchmark")
    totals = {}
    for policy in rows[0]["policies"]:
        arms = [row["policies"][policy] for row in rows]
        counts = {key: sum(arm[key] for arm in arms) for key in (
            "correct", "generation_calls", "generation_input_tokens", "generation_output_tokens", "prm_calls",
        )}
        totals[policy] = {**counts, "accuracy": counts["correct"] / len(rows),
                          "stop_reasons": dict(Counter(arm["reason"] for arm in arms))}
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                      for row in rows).encode("utf-8")
    result = {
        "samples": len(rows), "formal": sample is None, "method": configs[0]["extension"]["method"],
        "wall_seconds": wall_seconds, "inference_p50_seconds": median(row["inference_seconds"] for row in rows),
        "physical_prm_calls": sum(len(row["prm_indices"]) for row in rows),
        "physical_generation_calls": sum(len(row["calls"]) for row in rows), "policies": totals,
        "configs": configs, "executions": executions, "retries": retries,
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output = root / "export"
    output.mkdir(exist_ok=True)
    (output / "predictions.jsonl").write_bytes(payload)
    (output / "summary.json").write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"method": result["method"], "samples": len(rows), "policies": totals}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    audit(args.root, args.wall_seconds)

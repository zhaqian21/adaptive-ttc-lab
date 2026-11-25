"""Replay agreement thresholds through nine on the original E4 candidate pool."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_five import COUNTS, POOL_SHA, load_verified, paired, totals
from compare_four import encode_rows
from replay import HERE, SAMPLES, costs


TWO_COMMIT = "d978a946c874e5402ebead5e412153e0f568e0cd"
TWO_SHA = "0e18556e00cec8495ef837fccfd0ba1475b62c0540151b4f41427cb6a8299f1e"
ARMS = ("two", "three", "four", "five", "six", "seven", "eight", "nine")


def agreeing_prefix(candidates):
    first = candidates[0]["answer"]
    if not isinstance(first, str) or not first.strip():
        return 0
    answer = first.strip()
    for index, candidate in enumerate(candidates[1:], start=1):
        other = candidate["answer"]
        if not isinstance(other, str) or other.strip() != answer:
            return index
    return len(candidates)


def run(output):
    results = HERE / "results"
    pool = load_verified(results / "candidate_pool.jsonl", POOL_SHA)
    rows = load_verified(results / "two_predictions.jsonl", TWO_SHA)
    for index, (row, entry) in enumerate(zip(rows, pool)):
        if row["problem_id"] != index or (row["problem_id"], row["dataset_unique_id"]) != (
                entry["problem_id"], entry["dataset_unique_id"]):
            raise ValueError(f"Question coverage differs from the archived comparisons at row {index}")
        candidates = entry["candidates"]
        prefix = agreeing_prefix(candidates)
        row["agreeing_prefix"] = prefix
        for count, arm in enumerate(ARMS, start=2):
            stopped = prefix >= count
            if count <= 5:
                if stopped != row[f"{arm}_stopped"]:
                    raise ValueError(f"Existing {arm}-answer decision differs at question {index}")
                continue
            row[f"{arm}_stopped"] = stopped
            row[arm] = row["baseline"]
            if stopped:
                first = candidates[0]
                row[arm] = {"selected_iteration": 1, "answer": first["answer"], "correct": first["correct"],
                            **costs(candidates[:count], 0)}

    metrics = {arm: totals(rows, arm) for arm in ("baseline", *ARMS)}
    savings = {}
    for arm in ARMS:
        saved = {key: metrics["baseline"][key] - metrics[arm][key] for key in COUNTS}
        savings[arm] = {"saved": saved, "rate": {key: saved[key] / metrics["baseline"][key] for key in COUNTS}}
    payload = encode_rows(rows)
    report = {
        "samples": SAMPLES,
        "prior_comparison_commit": TWO_COMMIT,
        "candidate_pool_sha256": POOL_SHA,
        "prior_comparison_sha256": TWO_SHA,
        "comparison_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "thresholds": dict(zip(ARMS, range(2, 10))),
        "early_stops": {arm: sum(row[f"{arm}_stopped"] for row in rows) for arm in ARMS},
        "metrics": metrics,
        "savings_vs_baseline": savings,
        "paired_vs_baseline": {arm: paired(rows, "baseline", arm) for arm in ARMS},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "threshold_predictions.jsonl").write_bytes(payload)
    (output / "threshold_summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print("Setting     Correct     Early     Input+output tokens     PRM requests")
    for arm, metrics in result["metrics"].items():
        print(f"{arm:10}  {metrics['correct']:3}/{SAMPLES}      {result['early_stops'].get(arm, 0):3}           "
              f"{metrics['tokens']:>10,}           {metrics['prm_requests']:>5,}")

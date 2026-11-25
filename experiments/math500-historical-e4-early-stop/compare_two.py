"""Replay two-answer agreement on the fixed historical E4 candidate pool."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_five import COUNTS, POOL_SHA, load_verified, paired, totals
from compare_four import encode_rows
from replay import HERE, SAMPLES, costs


FOUR_COMMIT = "a8c40339fefbd72ad127f9f54f64cf27543c5cd7"
FOUR_SHA = "36f1fd532b1c18ef6938edb71c62c89b1d1682a52473f2123b3179ece547c47b"


def run(output):
    results = HERE / "results"
    pool = load_verified(results / "candidate_pool.jsonl", POOL_SHA)
    rows = load_verified(results / "four_predictions.jsonl", FOUR_SHA)

    for index, (row, item) in enumerate(zip(rows, pool)):
        if row["problem_id"] != index or (row["problem_id"], row["dataset_unique_id"]) != (
                item["problem_id"], item["dataset_unique_id"]):
            raise ValueError(f"Question coverage differs from the archived comparisons at row {index}")
        candidates = item["candidates"]
        first, second = [candidate["answer"] for candidate in candidates[:2]]
        stopped = (isinstance(first, str) and isinstance(second, str)
                   and bool(first.strip()) and first.strip() == second.strip())
        row["two_stopped"] = stopped
        row["two"] = row["baseline"]
        if stopped:
            row["two"] = {"selected_iteration": 1, "answer": first, "correct": candidates[0]["correct"],
                          **costs(candidates[:2], 0)}

    arms = ("baseline", "two", "three", "four", "five")
    metrics = {arm: totals(rows, arm) for arm in arms}
    saved = {key: metrics["baseline"][key] - metrics["two"][key] for key in COUNTS}
    payload = encode_rows(rows)
    report = {
        "samples": SAMPLES,
        "four_answer_replay_commit": FOUR_COMMIT,
        "candidate_pool_sha256": POOL_SHA,
        "prior_comparison_sha256": FOUR_SHA,
        "comparison_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "early_stops": {arm: sum(row[f"{arm}_stopped"] for row in rows) for arm in arms[1:]},
        "metrics": metrics,
        "two_savings_vs_baseline": {"saved": saved,
                                    "rate": {key: saved[key] / metrics["baseline"][key] for key in COUNTS}},
        "two_minus_three": {key: metrics["two"][key] - metrics["three"][key] for key in COUNTS},
        "two_vs_baseline_paired": paired(rows, "baseline", "two"),
        "two_vs_three_paired": paired(rows, "three", "two"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "two_predictions.jsonl").write_bytes(payload)
    (output / "two_summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))

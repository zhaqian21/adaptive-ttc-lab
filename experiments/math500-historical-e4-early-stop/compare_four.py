"""Replay four-answer agreement alongside the archived three- and five-answer runs."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_five import COUNTS, POOL_SHA, THREE_SHA, compare_row, load_verified, paired, totals
from replay import BASELINE, HERE, SAMPLES, costs, read_baseline


FIVE_SHA = "328cb3679d83f7ccfee7e4f6f9a5a68fc1c46e5986eb798933aee66b62f3405f"
FIVE_COMMIT = "dcc0b0fefa255a1fe298841f398917a97fbd5cc4"


def encode_rows(rows):
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                   for row in rows).encode("utf-8")


def run(output):
    baseline = read_baseline(BASELINE)
    pool = load_verified(HERE / "results" / "candidate_pool.jsonl", POOL_SHA)
    three = load_verified(HERE / "results" / "predictions.jsonl", THREE_SHA)
    rows = [compare_row(entry, ref, original) for entry, ref, original in zip(pool, baseline, three)]
    if hashlib.sha256(encode_rows(rows)).hexdigest() != FIVE_SHA:
        raise ValueError("Three- and five-answer replay failed to reproduce PR #15")

    for row, item in zip(rows, pool):
        first, _, _, fourth, _ = row["first_five_answers"]
        stopped = row["three_stopped"] and isinstance(fourth, str) and fourth.strip() == first.strip()
        row["four_stopped"] = stopped
        row["four"] = row["baseline"]
        if stopped:
            candidates = item["candidates"]
            row["four"] = {"selected_iteration": 1, "answer": first, "correct": candidates[0]["correct"],
                           **costs(candidates[:4], 0)}

    arms = ("baseline", "three", "four", "five")
    metrics = {name: totals(rows, name) for name in arms}
    saved = {key: metrics["baseline"][key] - metrics["four"][key] for key in COUNTS}
    payload = encode_rows(rows)
    report = {
        "samples": SAMPLES,
        "five_answer_replay_commit": FIVE_COMMIT,
        "candidate_pool_sha256": POOL_SHA,
        "prior_comparison_sha256": FIVE_SHA,
        "comparison_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "early_stops": {arm: sum(row[f"{arm}_stopped"] for row in rows) for arm in arms[1:]},
        "metrics": metrics,
        "four_savings_vs_baseline": {"saved": saved,
                                     "rate": {key: saved[key] / metrics["baseline"][key] for key in COUNTS}},
        "four_minus_three": {key: metrics["four"][key] - metrics["three"][key] for key in COUNTS},
        "four_minus_five": {key: metrics["four"][key] - metrics["five"][key] for key in COUNTS},
        "four_vs_baseline_paired": paired(rows, "baseline", "four"),
        "four_vs_three_paired": paired(rows, "three", "four"),
        "four_vs_five_paired": paired(rows, "five", "four"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "four_predictions.jsonl").write_bytes(payload)
    (output / "four_summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))

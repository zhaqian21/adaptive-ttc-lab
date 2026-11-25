"""Compare all 2–9 agreement thresholds on the historical E2 and E4 pools."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_five import COUNTS, POOL_SHA, load_verified, paired, totals
from compare_four import encode_rows
from compare_thresholds import ARMS, agreeing_prefix
from prepare_e2_pool import E2_BASELINE, E2_BASELINE_SHA
from replay import BASELINE, BASELINE_SHA, HERE, SAMPLES, costs, verified_baseline


E2_POOL_SHA = "97bbc5e3be2fa1fb654e853e497efb4606c173df9e43732a18ee57c4df345600"
E4_SWEEP_SHA = "866a1abccf124b8820f0b413b04217791c84db118667b251f923a8294658609e"


def sweep(pool, refs):
    rows = []
    for index, (entry, ref) in enumerate(zip(pool, refs)):
        if ref["problem_id"] != index:
            raise ValueError(f"Unexpected baseline question at row {index}")
        baseline = verified_baseline(entry, ref)
        candidates = entry["candidates"]
        prefix = agreeing_prefix(candidates)
        first = candidates[0]
        row = {
            "problem_id": index, "dataset_unique_id": entry["dataset_unique_id"],
            "agreeing_prefix": prefix, "baseline": baseline,
        }
        for count, arm in enumerate(ARMS, start=2):
            stopped = prefix >= count
            row[f"{arm}_stopped"] = stopped
            row[arm] = ({
                "selected_iteration": 1, "answer": first["answer"], "correct": first["correct"],
                **costs(candidates[:count], 0),
            } if stopped else baseline)
        rows.append(row)
    return rows


def summarize(rows, baseline_correct, baseline_calls):
    metrics = {arm: totals(rows, arm) for arm in ("baseline", *ARMS)}
    baseline = metrics["baseline"]
    if (baseline["correct"], baseline["generation_calls"], baseline["prm_requests"]) != (
            baseline_correct, baseline_calls, 5_000):
        raise ValueError("Historical baseline did not reconcile with the PR #2 audit")
    savings = {}
    for arm in ARMS:
        saved = {key: baseline[key] - metrics[arm][key] for key in COUNTS}
        savings[arm] = {"saved": saved, "rate": {key: saved[key] / baseline[key] for key in COUNTS}}
    return {
        "metrics": metrics,
        "early_stops": {arm: sum(row[f"{arm}_stopped"] for row in rows) for arm in ARMS},
        "savings_vs_baseline": savings,
        "paired_vs_baseline": {arm: paired(rows, "baseline", arm) for arm in ARMS},
    }


def run(output):
    results = HERE / "results"
    e2_rows = sweep(load_verified(results / "e2_candidate_pool.jsonl", E2_POOL_SHA),
                    load_verified(E2_BASELINE, E2_BASELINE_SHA))
    e4_rows = sweep(load_verified(results / "candidate_pool.jsonl", POOL_SHA),
                    load_verified(BASELINE, BASELINE_SHA))
    e4_archive = load_verified(results / "threshold_predictions.jsonl", E4_SWEEP_SHA)
    for direct, dynamic, archived in zip(e2_rows, e4_rows, e4_archive):
        if direct["dataset_unique_id"] != dynamic["dataset_unique_id"]:
            raise ValueError(f"E2 and E4 use different data on question {direct['problem_id']}")
        if any(archived[key] != value for key, value in dynamic.items()):
            raise ValueError(f"E4 replay differs from PR #18 on question {dynamic['problem_id']}")

    payload = encode_rows(e2_rows)
    report = {
        "samples": SAMPLES,
        "thresholds": dict(zip(ARMS, range(2, 10))),
        "source_sha256": {
            "e2_baseline_predictions": E2_BASELINE_SHA, "e2_candidate_pool": E2_POOL_SHA,
            "e4_baseline_predictions": BASELINE_SHA, "e4_candidate_pool": POOL_SHA,
            "e4_threshold_predictions": E4_SWEEP_SHA,
        },
        "e2_threshold_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "E2": summarize(e2_rows, baseline_correct=395, baseline_calls=10_000),
        "E4": summarize(e4_rows, baseline_correct=397, baseline_calls=108_845),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "e2_threshold_predictions.jsonl").write_bytes(payload)
    (output / "e2_e4_comparison.json").write_bytes(
        (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    return report


def print_table(report):
    print("| Rule | E2 accuracy | E2 tokens | E2 PRM requests | E4 accuracy | E4 tokens | E4 PRM requests |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in ("baseline", *ARMS):
        label = "K=10 baseline" if arm == "baseline" else f"K={report['thresholds'][arm]} agreement"
        cells = [label]
        for case in ("E2", "E4"):
            row = report[case]["metrics"][arm]
            cells.extend((f"{row['accuracy']:.1%}", f"{row['tokens']:,}", f"{row['prm_requests']:,}"))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print_table(run(args.output))

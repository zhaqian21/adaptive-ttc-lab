"""Compare three- and five-answer agreement on the fixed historical E4 pool."""

import argparse
import hashlib
import json
from pathlib import Path

from replay import BASELINE, HERE, METRICS, SAMPLES, costs, read_baseline, replay_row


POOL_SHA = "c1ece04f0e8d727a00bba995ab1320a849af25a7368608c0e3a64befca517373"
THREE_SHA = "03e4e6057f2800a0f975cbf8d8d68ccc378180b9597e3d07636c837667425297"
THREE_COMMIT = "8a9407db2de37709390ebdd6edff0d1fbef8a497"
COUNTS = (*METRICS, "tokens", "prm_requests")


def load_verified(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Expected the exact PR #14 archive at {path}")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != SAMPLES:
        raise ValueError(f"Expected {SAMPLES} rows at {path}")
    return rows


def compare_row(pool, reference, archived_three):
    three = replay_row(pool, reference)
    if three != archived_three:
        raise ValueError(f"Three-answer replay differs from PR #14 on question {three['problem_id']}")

    candidates = pool["candidates"]
    answers = [candidate["answer"] for candidate in candidates[:5]]
    early_three = three["stop_reason"] == "three_answer_agreement"
    early_five = early_three and all(
        isinstance(answer, str) and answer.strip() == answers[0].strip() for answer in answers[3:]
    )
    five = three["baseline"]
    if early_five:
        five = {"selected_iteration": 1, "answer": answers[0], "correct": candidates[0]["correct"],
                **costs(candidates[:5], 0)}

    return {
        "problem_id": three["problem_id"],
        "dataset_unique_id": three["dataset_unique_id"],
        "first_five_answers": answers,
        "three_stopped": early_three,
        "five_stopped": early_five,
        "baseline": three["baseline"],
        "three": three["replay"],
        "five": five,
    }


def totals(rows, variant):
    correct = sum(row[variant]["correct"] is True for row in rows)
    return {"correct": correct, "accuracy": correct / len(rows),
            **{name: sum(row[variant][name] for row in rows) for name in COUNTS}}


def paired(rows, original, variant):
    gained = [r["problem_id"] for r in rows if r[original]["correct"] is not True and r[variant]["correct"] is True]
    lost = [r["problem_id"] for r in rows if r[original]["correct"] is True and r[variant]["correct"] is not True]
    return {"gained": len(gained), "lost": len(lost), "gained_problem_ids": gained, "lost_problem_ids": lost}


def run(baseline_path, pool_path, three_path, output):
    baseline = read_baseline(baseline_path)
    pool = load_verified(pool_path, POOL_SHA)
    archived_three = load_verified(three_path, THREE_SHA)
    rows = [compare_row(entry, ref, archived) for entry, ref, archived in zip(pool, baseline, archived_three)]
    metrics = {arm: totals(rows, arm) for arm in ("baseline", "three", "five")}
    savings = {}
    for arm in ("three", "five"):
        saved = {key: metrics["baseline"][key] - metrics[arm][key] for key in COUNTS}
        savings[arm] = {"saved": saved,
                        "rate": {key: saved[key] / metrics["baseline"][key] for key in COUNTS}}
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in rows).encode("utf-8")
    report = {
        "samples": SAMPLES,
        "three_answer_replay_commit": THREE_COMMIT,
        "candidate_pool_sha256": POOL_SHA,
        "three_answer_predictions_sha256": THREE_SHA,
        "comparison_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "early_stops": {arm: sum(row[f"{arm}_stopped"] for row in rows) for arm in ("three", "five")},
        "metrics": metrics,
        "savings_vs_baseline": savings,
        "five_minus_three": {key: metrics["five"][key] - metrics["three"][key] for key in COUNTS},
        "five_vs_baseline_paired": paired(rows, "baseline", "five"),
        "five_vs_three_paired": paired(rows, "three", "five"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "five_predictions.jsonl").write_bytes(payload)
    (output / "five_summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--pool", type=Path, default=HERE / "results" / "candidate_pool.jsonl")
    parser.add_argument("--three", type=Path, default=HERE / "results" / "predictions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.baseline, args.pool, args.three, args.output), indent=2))

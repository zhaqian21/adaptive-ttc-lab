"""Replay three-answer agreement on the verified historical E4 candidate pool."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from prepare_pool import BASELINE_SHA, CANDIDATES, DATASET, SAMPLES, UPSTREAM, read_baseline


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "math500-published-code-2026-11-24" / "results" / "E4" / "predictions.jsonl"
METRICS = ("generation_calls", "input_tokens", "output_tokens")


def agree(candidates):
    answers = [candidate["answer"] for candidate in candidates[:3]]
    return (len(answers) == 3 and all(isinstance(a, str) and a.strip() for a in answers)
            and len({a.strip() for a in answers}) == 1)


def costs(candidates, prm_requests):
    result = {name: sum(candidate[name] for candidate in candidates) for name in METRICS}
    return {**result, "tokens": result["input_tokens"] + result["output_tokens"], "prm_requests": prm_requests}


def replay_row(pool, ref):
    candidates = pool["candidates"]
    index = ref["problem_id"]
    if ((pool["problem_id"], pool["dataset_unique_id"]) != (index, ref["dataset_unique_id"])
            or len(candidates) != CANDIDATES):
        raise ValueError(f"Wrong historical pool on question {index}")
    if any(not math.isfinite(c["mean_reward"]) or not 0 <= c["mean_reward"] <= 1
           or any(not isinstance(c[k], int) or c[k] <= 0 for k in METRICS) for c in candidates):
        raise ValueError(f"Invalid candidate costs or scores on question {index}")

    all_rewards = [c["mean_reward"] for c in candidates]
    selected = all_rewards.index(max(all_rewards)) if any(all_rewards) else CANDIDATES - 1
    historical = candidates[selected]
    base_cost = costs(candidates, CANDIDATES)
    if (pool["selected_iteration"], historical["answer"], historical["correct"],
            base_cost["output_tokens"], base_cost["generation_calls"]) != (
            ref["selected_iteration"], ref["predicted_answer"], ref["is_correct"], ref["output_tokens"], ref["llm_calls"]):
        raise ValueError(f"Could not reproduce historical E4 on question {index}")

    early = agree(candidates)
    chosen = candidates[0] if early else historical
    replay_cost = costs(candidates[:3], 0) if early else base_cost
    return {
        "problem_id": index,
        "dataset_unique_id": pool["dataset_unique_id"],
        "first_three_answers": [candidate["answer"] for candidate in candidates[:3]],
        "stop_reason": "three_answer_agreement" if early else "original_e4",
        "baseline": {"selected_iteration": selected + 1, "answer": historical["answer"],
                     "correct": historical["correct"], **base_cost},
        "replay": {"selected_iteration": 1 if early else selected + 1, "answer": chosen["answer"],
                   "correct": chosen["correct"], **replay_cost},
    }


def run(baseline_path, pool_path, output):
    baseline = read_baseline(baseline_path)
    raw = pool_path.read_bytes()
    pool = [json.loads(line) for line in raw.splitlines()]
    if len(pool) != SAMPLES:
        raise ValueError(f"Expected {SAMPLES} archived candidate lists")
    rows = [replay_row(row, ref) for row, ref in zip(pool, baseline)]
    metrics = (*METRICS, "tokens", "prm_requests")
    summary = {}
    for variant in ("baseline", "replay"):
        correct = sum(row[variant]["correct"] is True for row in rows)
        summary[variant] = {"correct": correct, "accuracy": correct / SAMPLES,
                            **{name: sum(row[variant][name] for row in rows) for name in metrics}}
    if (summary["baseline"]["correct"], summary["baseline"]["tokens"], summary["baseline"]["prm_requests"]) != (
            397, 72_861_820, 5000):
        raise ValueError("Historical baseline accuracy or costs did not reconcile")

    gained = sum(row["baseline"]["correct"] is not True and row["replay"]["correct"] is True for row in rows)
    lost = sum(row["baseline"]["correct"] is True and row["replay"]["correct"] is not True for row in rows)
    saved = {name: summary["baseline"][name] - summary["replay"][name] for name in metrics}
    rates = {name: saved[name] / summary["baseline"][name] for name in metrics}
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode()
    report = {
        "samples": SAMPLES, "baseline_commit": "9d51d691be6d58c0eb78fef0082d039a24182d97",
        "upstream_commit": UPSTREAM, "dataset_sha256": DATASET,
        "baseline_predictions_sha256": BASELINE_SHA, "candidate_pool_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "early_stops": sum(row["stop_reason"] == "three_answer_agreement" for row in rows),
        "paired_correctness": {"gained": gained, "lost": lost},
        **summary, "saved": saved, "saving_rate": rates,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_bytes(payload)
    (output / "summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--pool", type=Path, default=HERE / "results" / "candidate_pool.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.baseline, args.pool, args.output), indent=2))

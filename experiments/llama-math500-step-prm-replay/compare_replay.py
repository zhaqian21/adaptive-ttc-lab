"""Audit the new scores and compare both selectors on identical question sets."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from prepare_pool import (
    CASES, EXPECTED_CORRECT, K, PRM_PROMPT, PRM_REVISION, SAMPLES, STEP_MARKER,
    digest, encode, load_baseline, read_pool, winner,
)


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "llama-table1-24-2026-11-25" / "results" / "math"
PAPER_CORRECT = {"E2": 223, "E4": 327}


def read(path):
    return json.loads(path.read_bytes())


def read_scores(folder, case, input_sha):
    config = read(folder / "config.json")
    runtime = read(folder / "runtime.json")
    path = folder / "step_scores.jsonl"
    if ((config["case"], config["input_sha256"], config["prm_revision"], config["system_prompt"], config["marker"])
            != (case, input_sha, PRM_REVISION, PRM_PROMPT, STEP_MARKER)
            or runtime["candidates"] != SAMPLES * K or digest(path) != runtime["score_sha256"]):
        raise ValueError(f"{case}: scorer configuration or score digest")
    for name, expected in config["source_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"{case}: scorer source changed at {name}")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if [(row["problem_id"], row["iteration"]) for row in rows] != [
        (i, k) for i in range(SAMPLES) for k in range(1, K + 1)
    ]:
        raise ValueError(f"{case}: wrong coverage in archived step scores")
    return rows, config, runtime


def selected_record(candidates, scores):
    index = winner(scores)
    candidate = candidates[index - 1]
    return {"selected_iteration": index, "answer": candidate["answer"],
            "is_correct": candidate["correct"], "score": scores[index - 1]}


def compare_case(case, pool_path, baseline, score_folder):
    pool = read_pool(pool_path)
    input_sha = digest(pool_path)
    refs, baseline_sha = load_baseline(baseline, case)
    scores, config, runtime = read_scores(score_folder, case, input_sha)
    result = []
    histogram = Counter()
    totals = Counter()
    for ref, entry in zip(refs, pool):
        index = ref["problem_id"]
        if (entry["case"], entry["problem_id"], entry["dataset_unique_id"], entry["baseline_sha256"]) != (
                case, index, ref["dataset_unique_id"], baseline_sha):
            raise ValueError(f"{case}/{index}: pool does not match PR #20")
        candidates = entry["candidates"]
        hist_scores = [row["legacy_score"] for row in candidates]
        legacy = selected_record(candidates, hist_scores)
        if (legacy["selected_iteration"], legacy["answer"], legacy["is_correct"]) != (
                ref["selected_iteration"], ref["predicted_answer"], ref["is_correct"]):
            raise ValueError(f"{case}/{index}: legacy selector did not reproduce baseline")
        step_scores, step_counts = [], []
        for candidate, scored in zip(candidates, scores[index * K:(index + 1) * K]):
            rewards = scored["step_rewards"]
            if (len(rewards) != len(candidate["steps"]) or not rewards
                    or any(not math.isfinite(x) or not 0 <= x <= 1 for x in rewards)):
                raise ValueError(f"{case}/{index}/{candidate['iteration']}: invalid step scores")
            step_scores.append(mean(rewards))
            step_counts.append(len(rewards))
        step = selected_record(candidates, step_scores)
        a, b = legacy["is_correct"] is True, step["is_correct"] is True
        totals["legacy_correct"] += a
        totals["step_mean_correct"] += b
        totals["wrong_to_correct"] += not a and b
        totals["correct_to_wrong"] += a and not b
        totals["selected_iteration_changed"] += legacy["selected_iteration"] != step["selected_iteration"]
        totals["answer_changed"] += legacy["answer"] != step["answer"]
        totals["oracle_correct"] += any(candidate["correct"] is True for candidate in candidates)
        totals["candidates_with_multiple_steps"] += sum(count > 1 for count in step_counts)
        totals["steps"] += sum(step_counts)
        histogram.update(step_counts)

        result.append({
            "problem_id": index, "dataset_unique_id": entry["dataset_unique_id"],
            "historical_result_sha256": entry["historical_result_sha256"],
            "legacy": legacy, "step_mean": step,
            "candidates": [
                {"iteration": c["iteration"], "is_correct": c["correct"], "legacy_score": c["legacy_score"],
                 "steps": n, "step_mean_score": score}
                for c, n, score in zip(candidates, step_counts, step_scores)
            ],
        })

    if totals["legacy_correct"] != EXPECTED_CORRECT[case]:
        raise ValueError(f"{case}: wrong baseline accuracy")
    summary = {
        "samples": SAMPLES, "candidates": SAMPLES * K, **dict(totals),
        "legacy_accuracy": totals["legacy_correct"] / SAMPLES,
        "step_mean_accuracy": totals["step_mean_correct"] / SAMPLES,
        "change_pp": (totals["step_mean_correct"] - totals["legacy_correct"]) * 100 / SAMPLES,
        "paper_correct": PAPER_CORRECT[case],
        "legacy_minus_paper_pp": (totals["legacy_correct"] - PAPER_CORRECT[case]) * 100 / SAMPLES,
        "step_mean_minus_paper_pp": (totals["step_mean_correct"] - PAPER_CORRECT[case]) * 100 / SAMPLES,
        "step_count_histogram": dict(sorted(histogram.items())),
        "input_sha256": input_sha, "baseline_sha256": baseline_sha,
        "raw_scores_sha256": runtime["score_sha256"],
        "scorer": config, "runtime": runtime,
    }
    return result, summary


def across_cases(e2_rows, e4_rows, arm):
    counts = {"both_correct": 0, "only_e2_correct": 0, "only_e4_correct": 0, "neither_correct": 0}
    for e2, e4 in zip(e2_rows, e4_rows):
        if (e2["problem_id"], e2["dataset_unique_id"]) != (e4["problem_id"], e4["dataset_unique_id"]):
            raise ValueError("E2 and E4 question IDs differ")
        a, b = e2[arm]["is_correct"] is True, e4[arm]["is_correct"] is True
        key = {(True, True): "both_correct", (True, False): "only_e2_correct",
               (False, True): "only_e4_correct", (False, False): "neither_correct"}[(a, b)]
        counts[key] += 1
    counts["e2_minus_e4_pp"] = (counts["only_e2_correct"] - counts["only_e4_correct"]) * 100 / SAMPLES
    return counts


def run(inputs, baseline, score_root, output):
    output.mkdir(parents=True, exist_ok=True)
    all_rows, results = {}, {}
    for case in CASES:
        label = case.lower()
        rows, summary = compare_case(case, inputs / f"{label}_pool.jsonl.gz", baseline / case, score_root / label)
        path = output / f"{label}_paired.jsonl"
        path.write_bytes(b"".join(encode(row) for row in rows))
        summary["paired_sha256"] = digest(path)
        all_rows[case] = rows
        results[case] = summary
    summary = {
        "cases": results,
        "paired_between_methods": {
            arm: across_cases(all_rows["E2"], all_rows["E4"], arm) for arm in ("legacy", "step_mean")
        },
    }
    (output / "summary.json").write_bytes((json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return summary


def markdown_table(report):
    lines = [
        "| 设置 | 历史整段 | 逐步均值 | 变化（百分点） | 改对 | 改错 | 换选轨迹 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, row in report["cases"].items():
        lines.append(
            f"| {case} | {row['legacy_correct']}/{SAMPLES} ({row['legacy_accuracy']:.1%}) | "
            f"{row['step_mean_correct']}/{SAMPLES} ({row['step_mean_accuracy']:.1%}) | {row['change_pp']:+.1f} | "
            f"{row['wrong_to_correct']} | {row['correct_to_wrong']} | {row['selected_iteration_changed']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--scores", type=Path, default=HERE / "results" / "scores")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(markdown_table(run(args.inputs, args.baseline, args.scores, args.output)))

"""Replay dynamic K=3 or K=5 with the original E4 PRM rule."""

import argparse
import hashlib
import json
from pathlib import Path

from load_pools import K, PREFIX, SMALL_K, answer_key, chosen, load, prefix_agrees
from replay_selection import contrast, encode


MODES = ("fixed3", "fixed5", "fixed10", "early5_to_10", "dynamic3_to_5")


def raw_prm(candidates):
    scores = [c["mean_reward"] for c in candidates]
    return scores.index(max(scores)) + 1 if any(scores) else len(scores)


def outcome(row, iteration, count, requests):
    return {**chosen(row["candidates"], iteration), "k": count, "prm_requests": requests,
            "tokens": row[f"tokens{count}"]}


def replay(row):
    candidates = row["candidates"]
    first_three = candidates[:SMALL_K]
    keys = [answer_key(c["predicted_answer"]) for c in first_three]
    stop3 = keys[0] is not None and len(set(keys)) == 1

    fixed3 = outcome(row, raw_prm(first_three), SMALL_K, SMALL_K)
    fixed5 = outcome(row, raw_prm(candidates[:PREFIX]), PREFIX, PREFIX)
    fixed10 = outcome(row, row["original_iteration"], K, K)
    early5 = outcome(row, 1, PREFIX, 0) if prefix_agrees(candidates) else fixed10
    dynamic = outcome(row, 1, SMALL_K, 0) if stop3 else fixed5
    return {"problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"],
            "fixed3": fixed3, "fixed5": fixed5, "fixed10": fixed10,
            "early5_to_10": early5, "dynamic3_to_5": dynamic}


def summarize(rows):
    report = {"samples": len(rows), "methods": {}}
    for mode in MODES:
        values = [r[mode] for r in rows]
        k3 = sum(v["k"] == SMALL_K for v in values)
        k5 = sum(v["k"] == PREFIX for v in values)
        report["methods"][mode] = {
            "correct": sum(v["is_correct"] for v in values),
            "tokens": sum(v["tokens"] for v in values),
            "prm_requests": sum(v["prm_requests"] for v in values),
            "candidates": sum(v["k"] for v in values),
            "k3": k3, "k5": k5, "k10": len(values) - k3 - k5,
        }
    report["dynamic_vs_fixed5"] = contrast(rows, "fixed5", "dynamic3_to_5")
    report["dynamic_vs_fixed10"] = contrast(rows, "fixed10", "dynamic3_to_5")
    return report


def run(qwen_pool, output):
    pools = load(qwen_pool)
    report = {"method": "three_unanimous_else_five_using_original_prm", "datasets": {}}
    output.mkdir(parents=True, exist_ok=True)
    for name, pool in pools.items():
        rows = [replay(row) for row in pool]
        payload = b"".join(encode(row) for row in rows)
        (output / f"{name}.jsonl").write_bytes(payload)
        report["datasets"][name] = {**summarize(rows), "predictions_sha256": hashlib.sha256(payload).hexdigest()}
    (output / "summary.json").write_bytes(
        (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-pool", type=Path, required=True, help="PR #15 candidate_pool.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.qwen_pool, args.output)
    for study, content in report["datasets"].items():
        print(study, content["samples"], *[content["methods"][method]["correct"] for method in MODES],
              "k3 stops", content["methods"]["dynamic3_to_5"]["k3"])

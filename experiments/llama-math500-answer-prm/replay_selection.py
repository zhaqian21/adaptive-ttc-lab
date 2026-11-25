"""Replay answer filtering and the PR #8 grouped PRM selector on fixed Llama pools."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
E2 = EXPERIMENTS / "llama-e2-generation-backends" / "results"
E4_POOL = EXPERIMENTS / "llama-math500-step-prm-replay" / "inputs" / "e4_pool.jsonl.gz"
E4_REFERENCE = EXPERIMENTS / "llama-table1-24-2026-11-25" / "results" / "math" / "E4" / "predictions.jsonl"
E4_SHA = "73c3f95e4ad26e7703c6068de73547dc4fe9b1bca23332254dd7e357c408ec3d"
E4_BASELINE_SHA = "ea19fc37fd3a4539f37f2f7f0c5e5da0070e14224467f604e07728ef6540d572"
E2_SHA = {
    "transformers": "9d91bedb62817ec5bc5b3aa9e9fee5adb315fdf7da0fcf727596659913eb18e8",
    "vllm": "c6c05ba44a03b47ce6697541a4c8eec228cd2d0d3e62ca263adf6e381182784a",
}
LABELS = {
    "e2_transformers": "E2 Transformers",
    "e2_vllm": "E2 vLLM",
    "e4": "历史 E4",
}
METHODS = ("original", "valid_max", "sum_prm")
K = 10


def encode(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def verify_bytes(path, expected_sha):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"Unexpected source checksum at {path}")
    return payload


def answer_key(answer):
    return answer.strip() or None if isinstance(answer, str) else None


def group_answers(predictions, rewards):
    """Reuse the grouping and ranking rules from PR #8."""
    members = {}
    for index, prediction in enumerate(predictions):
        answer = answer_key(prediction)
        if answer is not None:
            members.setdefault(answer, []).append(index)
    groups = []
    for answer, indices in members.items():
        best_index = max(indices, key=rewards.__getitem__)
        groups.append({
            "answer": answer, "members": [i + 1 for i in indices], "support": len(indices),
            "sum_prm": math.fsum(rewards[i] for i in indices), "best_reward": rewards[best_index],
            "representative": best_index + 1,
        })
    return groups


def selectors(candidates):
    if [r["iteration"] for r in candidates] != list(range(1, K + 1)):
        raise ValueError("Expected 10 candidates in original generation order")
    predictions = [r["predicted_answer"] for r in candidates]
    rewards = [r["mean_reward"] for r in candidates]
    if (any(not math.isfinite(score) or not 0 <= score <= 1 for score in rewards)
            or any(type(r["is_correct"]) not in (bool, type(None)) for r in candidates)):
        raise ValueError("Invalid historical PRM score or grade")
    original = rewards.index(max(rewards)) + 1 if any(rewards) else K
    valid = [i for i, prediction in enumerate(predictions) if answer_key(prediction) is not None]
    filtered = max(valid, key=rewards.__getitem__) + 1 if valid else original
    groups = group_answers(predictions, rewards)
    outcome = original
    if groups and any(group["sum_prm"] > 0 for group in groups):
        best = max(groups, key=lambda group: (group["sum_prm"], group["best_reward"], -group["representative"]))
        outcome = best["representative"]
    return dict(zip(METHODS, (original, filtered, outcome))), groups


def chosen(candidates, iteration):
    row = candidates[iteration - 1]
    return {
        "selected_iteration": iteration, "predicted_answer": row["predicted_answer"],
        "is_correct": row["is_correct"] is True, "mean_reward": row["mean_reward"],
    }


def load_e4(pool_path=E4_POOL, baseline_path=E4_REFERENCE):
    pool = [json.loads(line) for line in gzip.decompress(verify_bytes(pool_path, E4_SHA)).splitlines()]
    refs = [json.loads(line) for line in verify_bytes(baseline_path, E4_BASELINE_SHA).splitlines()]
    if len(pool) != 500 or len(refs) != 500:
        raise ValueError("Expected both 500-question E4 archives")
    rows = []
    for index, (row, ref) in enumerate(zip(pool, refs)):
        if ((row["problem_id"], row["dataset_unique_id"]) != (index, ref["dataset_unique_id"])
                or ref["problem_id"] != index):
            raise ValueError(f"E4 archive mismatch at question {index}")
        candidates = [
            {"iteration": c["iteration"], "predicted_answer": c["answer"],
             "is_correct": c["correct"], "mean_reward": c["legacy_score"]}
            for c in row["candidates"]
        ]
        rows.append(({
            "problem_id": index, "dataset_unique_id": row["dataset_unique_id"],
            "historical_result_sha256": row["historical_result_sha256"],
        }, candidates, ref))
    return rows


def load_e2(backend):
    path = E2 / "pilot" / backend / "scored.jsonl"
    scored = [json.loads(line) for line in verify_bytes(path, E2_SHA[backend]).splitlines()]
    published = [json.loads(line) for line in (E2 / "paired.jsonl").read_bytes().splitlines()]
    if len(published) != 100 or len(scored) != K * len(published):
        raise ValueError("Expected the PR #23 100-question comparison")
    results = []
    for index, ref in enumerate(published):
        candidates = scored[index * K:(index + 1) * K]
        if any(row["problem_id"] != ref["problem_id"] for row in candidates):
            raise ValueError(f"{backend}: mismatched candidate IDs")
        anchor = ref[backend]
        results.append(({
            "problem_id": ref["problem_id"], "dataset_unique_id": ref["dataset_unique_id"],
        }, candidates, {**anchor, "is_correct": anchor["selected_correct"]}))
    return results


def contrast(rows, left, right):
    gained = [row["problem_id"] for row in rows if not row[left]["is_correct"] and row[right]["is_correct"]]
    lost = [row["problem_id"] for row in rows if row[left]["is_correct"] and not row[right]["is_correct"]]
    return {"wrong_to_correct": len(gained), "correct_to_wrong": len(lost), "net": len(gained) - len(lost),
            "gained_problem_ids": gained, "lost_problem_ids": lost}


def replay_case(inputs):
    rows = []
    for identity, candidates, ref in inputs:
        selections, groups = selectors(candidates)
        choices = {name: chosen(candidates, index) for name, index in selections.items()}
        original = choices["original"]
        if (original["selected_iteration"], original["predicted_answer"], original["is_correct"]) != (
                ref["selected_iteration"], ref["predicted_answer"], ref["is_correct"] is True):
            raise ValueError(f"Original selection mismatch on question {identity['problem_id']}")
        rows.append({
            **identity, "candidate_pool_has_correct_answer": any(c["is_correct"] is True for c in candidates),
            "answer_groups": groups, **choices,
        })
    return rows


def run(output):
    pools = {"e2_transformers": load_e2("transformers"), "e2_vllm": load_e2("vllm"), "e4": load_e4()}
    report = {"source_sha256": {**E2_SHA, "e4_pool": E4_SHA, "e4_reference": E4_BASELINE_SHA}, "cases": {}}
    output.mkdir(parents=True, exist_ok=True)
    for case, inputs in pools.items():
        rows = replay_case(inputs)
        payload = b"".join(encode(row) for row in rows)
        (output / f"{case}_paired.jsonl").write_bytes(payload)
        totals = {name: sum(row[name]["is_correct"] for row in rows) for name in METHODS}
        report["cases"][case] = {
            "samples": len(rows), "correct": totals, "accuracy": {name: n / len(rows) for name, n in totals.items()},
            "candidate_pool_correct": sum(row["candidate_pool_has_correct_answer"] for row in rows),
            "contrasts": {
                f"{left}_to_{right}": contrast(rows, left, right)
                for left, right in (("original", "valid_max"), ("original", "sum_prm"), ("valid_max", "sum_prm"))
            },
            "original_chosen_with_empty_answer": sum(answer_key(row["original"]["predicted_answer"]) is None for row in rows),
            "paired_sha256": hashlib.sha256(payload).hexdigest(),
        }
    (output / "summary.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


def markdown_table(report):
    lines = ["| 候选池 | 样本数 | 原单条最高分 | 过滤空答案 | 答案级 PRM 聚合 |",
             "|---|---:|---:|---:|---:|"]
    for case, metrics in report["cases"].items():
        values = " | ".join(f"{metrics['correct'][name]}（{metrics['accuracy'][name]:.1%}）" for name in METHODS)
        lines.append(f"| {LABELS[case]} | {metrics['samples']} | {values} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(markdown_table(run(parser.parse_args().output)))

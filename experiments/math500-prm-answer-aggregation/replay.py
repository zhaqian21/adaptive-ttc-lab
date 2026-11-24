"""Replay answer-level voting on the pinned PR #7 MATH-500 predictions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from answer_aggregation import answer_key, group_answers, winner


SOURCE_COMMIT = "135d4acc8379ad5f3f1e2a4cab47aec7bdbd982f"
SOURCE_PREDICTIONS_SHA256 = "fdc32aeeb9d6f58e50c5b061b40ad708917461af75b796091f6defef396ad8be"
SOURCE_SUMMARY_SHA256 = "c9b2bee9ed4f9815c05f8877595e3fa91e26b09ad3174e1301280e1ca2eabc59"
SOURCE_DIR = Path(__file__).resolve().parents[1] / "math500-adaptive-k" / "results" / "full"
METHODS = ("adaptive", "sum_prm", "majority")
COST_FIELDS = ("generation_input_tokens", "generation_output_tokens", "generation_calls", "prm_calls")


def replay_row(row):
    original = row["policies"]["adaptive"]
    count = original["candidates"]
    if not isinstance(count, int) or not 0 <= count <= len(row["candidates"]):
        raise ValueError("Invalid adaptive candidate count")
    candidates = row["candidates"][:count]
    predictions = [candidate["prediction"] for candidate in candidates]
    selected = original["selected_index"]
    route = original["reason"]
    indices = {name: selected for name in METHODS}
    groups = []
    if route == "agreement":
        if (count != 3 or selected != 0 or original["prm_calls"]
                or not answer_key(predictions[0]) or len(set(map(answer_key, predictions))) != 1):
            raise ValueError("Invalid original early-stop record")
    elif route in ("token_budget", "no_initial_candidates"):
        if original["prm_calls"] != count:
            raise ValueError("PRM calls do not match the adaptive prefix")
        rewards = [candidate["mean_reward"] for candidate in candidates]
        if not all(isinstance(reward, (int, float)) and 0 <= reward <= 1 for reward in rewards):
            raise ValueError("Invalid PRM scores")
        expected = max(range(count), key=rewards.__getitem__) if any(rewards) else (count - 1 if count else None)
        if selected != expected:
            raise ValueError("Original PRM selection did not match the archived scores")
        groups = group_answers(predictions, rewards)
        indices.update(sum_prm=winner(groups, "sum_prm", selected),
                       majority=winner(groups, "support", selected))
    else:
        raise ValueError(f"Unexpected adaptive route: {route}")

    choices = {}
    for name, index in indices.items():
        candidate = candidates[index] if index is not None else {"prediction": None, "correct": False}
        if type(candidate["correct"]) is not bool:
            raise ValueError("Missing candidate correctness label")
        choices[name] = {"selected_index": index, "prediction": candidate["prediction"], "correct": candidate["correct"]}
    if any(choices["adaptive"][key] != original[key] for key in choices["adaptive"]):
        raise ValueError("Reproduced adaptive answer differs from the original record")

    return {
        "problem_id": row["problem_id"],
        "dataset_unique_id": row["dataset_unique_id"],
        "reason": route,
        "prefix_candidates": count,
        "cost": {key: original[key] for key in COST_FIELDS},
        "candidate_pool_has_correct_answer": any(candidate["correct"] for candidate in candidates),
        "answer_groups": groups,
        "policies": choices,
    }


def source_data(directory):
    records = (directory / "predictions.jsonl").read_bytes()
    manifest = (directory / "summary.json").read_bytes()
    if (hashlib.sha256(records).hexdigest() != SOURCE_PREDICTIONS_SHA256
            or hashlib.sha256(manifest).hexdigest() != SOURCE_SUMMARY_SHA256):
        raise ValueError("Input differs from the pinned PR #7 MATH-500 audit")
    summary = json.loads(manifest)
    rows = [json.loads(line) for line in records.splitlines()]
    if (not summary["formal"] or summary["mode"] != "paired" or summary["samples"] != 500
            or summary["predictions_sha256"] != SOURCE_PREDICTIONS_SHA256
            or [row["problem_id"] for row in rows] != list(range(500))
            or len({row["dataset_unique_id"] for row in rows}) != 500):
        raise ValueError("Input is not the complete audited MATH-500 paired run")
    return rows, summary


def run(source, output):
    original_rows, original_summary = source_data(source)
    rows = [replay_row(row) for row in original_rows]
    total_cost = {key: sum(row["cost"][key] for row in rows) for key in COST_FIELDS}
    early = [row for row in rows if row["reason"] == "agreement"]
    routed = [row for row in rows if row["cost"]["prm_calls"]]
    original = original_summary["policies"]["adaptive"]
    if any(total_cost[key] != original[key] for key in total_cost) or dict(
        Counter(row["reason"] for row in rows)
    ) != original["stop_reasons"]:
        raise ValueError("Adaptive paths or costs differ from the original audit")

    policies = {}
    for name in METHODS:
        correct = sum(row["policies"][name]["correct"] for row in rows)
        pairs = Counter((row["policies"]["adaptive"]["correct"], row["policies"][name]["correct"]) for row in rows)
        policies[name] = {
            "correct": correct,
            "accuracy": correct / len(rows),
            "early_correct": sum(row["policies"][name]["correct"] for row in early),
            "routed_correct": sum(row["policies"][name]["correct"] for row in routed),
            "answers_changed": sum(row["policies"][name]["prediction"] != row["policies"]["adaptive"]["prediction"]
                                   for row in routed),
            "gained_vs_adaptive": pairs[(False, True)],
            "lost_vs_adaptive": pairs[(True, False)],
            **total_cost,
        }
    if policies["adaptive"]["correct"] != original["correct"]:
        raise ValueError("Replayed baseline accuracy does not match the audit")

    reference = original_summary["policies"]["full"]
    for key in ("correct", *COST_FIELDS):
        if sum(row["policies"]["full"][key] for row in original_rows) != reference[key]:
            raise ValueError("Full-resampling reference differs from its audit")
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                      for row in rows).encode("utf-8")
    result = {
        "source_commit": SOURCE_COMMIT,
        "source_predictions_sha256": SOURCE_PREDICTIONS_SHA256,
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "samples": len(rows),
        "early_stops": len(early),
        "routed_to_prm": len(routed),
        "unscored_without_candidates": len(rows) - len(early) - len(routed),
        "routed_with_correct_candidate": sum(row["candidate_pool_has_correct_answer"] for row in routed),
        "policies": policies,
        "full_resampling_reference": {key: reference[key] for key in ("correct", "accuracy", *COST_FIELDS)},
        "additional_generation_calls": 0,
        "additional_prm_calls": 0,
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_bytes(payload)
    (output / "summary.json").write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR, help="Directory containing the two pinned input files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("The replay output directory must be separate from the original input")
    print(json.dumps(run(args.source, args.output), ensure_ascii=False, indent=2))

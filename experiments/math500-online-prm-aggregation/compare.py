"""Compare two independently timed MATH-500 adaptive inference runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from sampling import Prefix, SELECTORS, select_adaptive


RUN_NAMES = dict(zip(SELECTORS, ("single-prm", "answer-prm-sum")))
COUNT_FIELDS = ("correct", "generation_input_tokens", "generation_output_tokens",
                "generation_calls", "prm_calls")


def load_run(root, selector):
    source = root / RUN_NAMES[selector] / "export"
    summary = json.loads((source / "summary.json").read_bytes())
    payload = (source / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    if (hashlib.sha256(payload).hexdigest() != summary["predictions_sha256"]
            or summary["mode"] != "adaptive" or len(rows) != summary["samples"]
            or summary["wall_seconds"] < 0
            or len({row["dataset_unique_id"] for row in rows}) != len(rows)
            or [row["problem_id"] for row in rows] != sorted({row["problem_id"] for row in rows})
            or (summary["formal"] and [row["problem_id"] for row in rows] != list(range(500)))):
        raise ValueError(f"Invalid audited run for {selector}")
    if (len(summary["configs"]) != 2 or len(summary["executions"]) != 2
            or any(config["selector"] != selector for config in summary["configs"])
            or any(e["failures"] for e in summary["executions"])):
        raise ValueError(f"Invalid shard configuration for {selector}")
    original = summary["policies"]["adaptive"]
    if (any(sum(row["policies"]["adaptive"][k] for row in rows) != original[k] for k in COUNT_FIELDS)
            or summary["physical_generation_calls"] != original["generation_calls"]
            or summary["physical_prm_calls"] != original["prm_calls"]):
        raise ValueError(f"Incorrect online costs for {selector}")
    elapsed = sorted(row["inference_seconds"] for row in rows)
    if any(not math.isfinite(value) or value < 0 for value in elapsed):
        raise ValueError("Invalid per-question inference time")
    metrics = {
        **original, "wall_seconds": summary["wall_seconds"], "shard_seconds": [e["seconds"] for e in summary["executions"]],
        "inference_p50_seconds": median(elapsed), "inference_p95_seconds": elapsed[math.ceil(0.95 * len(rows)) - 1],
        "input_predictions_sha256": summary["predictions_sha256"], "retries": summary["retries"],
    }
    return rows, summary, metrics


def choices(row):
    saved = row["policies"]["adaptive"]
    prefix = Prefix(saved["candidates"], saved["calls"], saved["reason"])
    results = {}
    for selector in SELECTORS:
        index = select_adaptive(row["candidates"], prefix, selector)
        item = row["candidates"][index] if index is not None else {"prediction": None, "correct": False}
        results[selector] = {"selected_index": index, "prediction": item["prediction"], "correct": item["correct"]}
    return results


def gain_loss(before, after):
    return {"gained": sum(not a and b for a, b in zip(before, after)),
            "lost": sum(a and not b for a, b in zip(before, after))}


def compare(root):
    single, grouped = SELECTORS
    runs = {name: load_run(root, name) for name in SELECTORS}
    first, second = runs[single][1], runs[grouped][1]
    for a, b in zip(first["configs"], second["configs"]):
        if {k: v for k, v in a.items() if k != "selector"} != {k: v for k, v in b.items() if k != "selector"}:
            raise ValueError("Independent runs have differing settings besides the selector")
    if first["formal"] != second["formal"] or first["samples"] != second["samples"]:
        raise ValueError("Run coverage differs")

    pairs = []
    for a, b in zip(runs[single][0], runs[grouped][0]):
        if (a["problem_id"], a["dataset_unique_id"]) != (b["problem_id"], b["dataset_unique_id"]):
            raise ValueError("Different items in the two runs")
        banks = {single: a, grouped: b}
        counterfactual = {name: choices(row) for name, row in banks.items()}
        for name, row in banks.items():
            saved = row["policies"]["adaptive"]
            if any(saved[k] != v for k, v in counterfactual[name][name].items()):
                raise ValueError(f"Archived online selection differs on question {row['problem_id']}")
        same_candidates = [x["prediction"] for x in a["candidates"]] == [x["prediction"] for x in b["candidates"]]
        same_rewards = same_candidates and [x["mean_reward"] for x in a["candidates"]] == [
            x["mean_reward"] for x in b["candidates"]]
        same_trace = same_rewards and all(a[k] == b[k] for k in ("calls", "attempts", "budget_boundary"))
        pairs.append({
            "problem_id": a["problem_id"], "dataset_unique_id": a["dataset_unique_id"],
            "same_stop_reason": a["policies"]["adaptive"]["reason"] == b["policies"]["adaptive"]["reason"],
            "same_candidate_answers": same_candidates, "same_candidate_answers_and_rewards": same_rewards,
            "same_trace": same_trace, "selections_on_each_candidate_pool": counterfactual,
        })

    def correctness(pool, selector):
        return [row["selections_on_each_candidate_pool"][pool][selector]["correct"] for row in pairs]

    by_pool = {}
    for name in SELECTORS:
        before, after = correctness(name, single), correctness(name, grouped)
        by_pool[name] = {single: sum(before), grouped: sum(after), **gain_loss(before, after)}
    run_delta = gain_loss(correctness(single, single), correctness(grouped, grouped))
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in pairs).encode("utf-8")
    report = {
        "samples": len(pairs), "formal": first["formal"],
        "runs": {name: runs[name][2] for name in SELECTORS}, "online_paired_gain_loss": run_delta,
        "same_candidate_pool_replay": by_pool,
        "matching_questions": {key: sum(row[key] for row in pairs) for key in (
            "same_stop_reason", "same_candidate_answers", "same_candidate_answers_and_rewards", "same_trace",
        )},
        "comparison_predictions_sha256": hashlib.sha256(payload).hexdigest(),
    }
    destination = root / "comparison"
    destination.mkdir(exist_ok=True)
    (destination / "predictions.jsonl").write_bytes(payload)
    (destination / "summary.json").write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(compare(parser.parse_args().root), ensure_ascii=False, indent=2))

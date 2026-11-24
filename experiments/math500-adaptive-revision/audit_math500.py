"""Validate both MATH-500 shards and export results paired with the archived E4 run."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from revision_policy import POLICIES


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def shard_files(source, shard):
    config = load(source / "config.json")
    execution = load(source / "execution.json")
    expected = config["question_ids"]
    failures = {int(index) for index in execution["failures"]}
    files = {int(path.stem): path for path in (source / "problems").glob("*.json")}
    if (config["shard"] != shard or config["shards"] != 2 or len(set(expected)) != len(expected)
            or execution["requested"] != len(expected) or execution["complete"] != len(files)
            or set(files) & failures or set(files) | failures != set(expected)
            or any(index % 2 != shard for index in expected)):
        raise ValueError(f"Incomplete or inconsistent shard {shard}: {source}")
    return config, execution, files


def comparable_retry_config(config):
    return {
        **{key: value for key, value in config.items() if key not in ("question_ids", "source_sha256")},
        "source_sha256": {key: value for key, value in config["source_sha256"].items()
                           if key != "run_math500.py"},
    }


def validate_result(row, config, index, dataset_unique_id):
    if (row["problem_id"] != index or row["dataset_unique_id"] != dataset_unique_id
            or set(row["policies"]) != set(POLICIES) or not row["initial_actions"]
            or len(row["initial_predictions"]) > config["initial_roots"]
            or sum(event["generation_tokens"] for event in row["initial_actions"]) != row["initial_tokens"]):
        raise ValueError(f"Invalid result for row {index}")
    for run in row["policies"].values():
        cost = run["generation_input_tokens"] + run["generation_output_tokens"]
        if (cost > config["token_budget"] or cost != row["initial_tokens"] +
                sum(event["generation_tokens"] for event in run["actions"])
                or run["prm_calls"] != len(run["candidates"]) or run["roots"] != len(run["candidates"])):
            raise ValueError(f"Invalid budget for row {index}")
        if not row["initial_predictions"]:
            if (run["roots"] or run["correct"] or run["prediction"] is not None
                    or run["selected_root"] is not None or run["actions"]):
                raise ValueError(f"Unanswered row {index} contains a selected candidate")
            continue
        rewards = [candidate["mean_reward"] for candidate in run["candidates"]]
        expected_selection = rewards.index(max(rewards)) if any(rewards) else len(rewards) - 1
        if not rewards or run["selected_root"] != expected_selection:
            raise ValueError(f"Invalid PRM selection for row {index}")
        selected = run["candidates"][expected_selection]
        if selected["correct"] != run["correct"] or selected["prediction"] != run["prediction"]:
            raise ValueError(f"Invalid selected answer for row {index}")


def summarize_policies(sample):
    summaries = {}
    for policy in POLICIES:
        runs = [row["policies"][policy] for row in sample]
        actions = [event for run in runs for event in run["actions"]]
        revisions = [event for event in actions if event["outcome"] == "revised"]
        relative = Counter((row["policies"]["resample"]["correct"], row["policies"][policy]["correct"])
                           for row in sample)
        summaries[policy] = {
            "correct": sum(run["correct"] for run in runs),
            "accuracy": mean(run["correct"] for run in runs),
            "mean_roots": mean(run["roots"] for run in runs),
            "generation_input_tokens": sum(run["generation_input_tokens"] for run in runs),
            "generation_output_tokens": sum(run["generation_output_tokens"] for run in runs),
            "generation_calls": sum(run["generation_calls"] for run in runs),
            "prm_calls": sum(run["prm_calls"] for run in runs),
            "prm_request_seconds": sum(run["prm_request_seconds"] for run in runs),
            "mean_question_seconds_in_mixed_run": mean(row["initial_seconds"] +
                                                       row["policies"][policy]["additional_seconds"] for row in sample),
            "actions": dict(Counter(event["outcome"] for event in actions)),
            "candidate_wrong_to_right": sum(not event["before_correct"] and event["after_correct"] for event in revisions),
            "candidate_right_to_wrong": sum(event["before_correct"] and not event["after_correct"] for event in revisions),
            "questions_wrong_to_right_vs_resample": relative[(False, True)],
            "questions_right_to_wrong_vs_resample": relative[(True, False)],
        }
    return summaries


def audit(root, baseline_path, destination, repair_root=None):
    baseline_rows = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines()]
    baseline = {row["problem_id"]: row for row in baseline_rows}
    if len(baseline_rows) != 500 or set(baseline) != set(range(500)):
        raise ValueError("Expected the complete archived 500-row E4 regression reference")

    configs, executions, repairs, rows = [], [], [], {}
    for shard in range(2):
        source = root / f"shard-{shard}"
        config, execution, files = shard_files(source, shard)
        repaired_ids = {int(index) for index in execution["failures"]}
        if repaired_ids:
            if repair_root is None:
                raise ValueError(f"Shard {shard} has unresolved rows {sorted(repaired_ids)}")
            repaired_config, repaired_execution, repaired_files = shard_files(repair_root / f"shard-{shard}", shard)
            if (repaired_execution["failures"] or set(repaired_files) != repaired_ids
                    or comparable_retry_config(config) != comparable_retry_config(repaired_config)):
                raise ValueError(f"Repair must cover exactly the failed IDs under the same configuration for shard {shard}")
            files.update(repaired_files)
            repairs.append({"shard": shard, "first_pass_failures": execution["failures"],
                            "config": repaired_config, "execution": repaired_execution})
        configs.append(config)
        executions.append(execution)
        for index, path in sorted(files.items()):
            if index in rows:
                raise ValueError(f"Invalid shard assignment for row {index}")
            row = load(path)
            reference = baseline[index]
            validate_result(row, config, index, reference["dataset_unique_id"])
            row["e4_reference_correct"] = reference["is_correct"] is True
            row["e4_reference_output_tokens"] = reference["output_tokens"]
            if index in repaired_ids:
                row["source"] = "repair"
            rows[index] = row

    ignored = {"shard", "question_ids"}
    if {key: value for key, value in configs[0].items() if key not in ignored} != {
        key: value for key, value in configs[1].items() if key not in ignored
    }:
        raise ValueError("Shard configurations differ")
    sample = [rows[index] for index in sorted(rows)]
    if not sample:
        raise ValueError("No audited rows")

    summaries = summarize_policies(sample)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = destination / "predictions.jsonl"
    predictions.write_bytes("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                   for row in sample).encode("utf-8"))
    summary = {
        "samples": len(sample), "formal": len(sample) == 500,
        "baseline_e4_correct_on_same_rows": sum(row["e4_reference_correct"] for row in sample),
        "baseline_e4_output_tokens_on_same_rows": sum(row["e4_reference_output_tokens"] for row in sample),
        "maximum_shard_execution_seconds": max(row["seconds"] for row in executions),
        "shard_counts": [len(config["question_ids"]) for config in configs],
        "repair_attempts": repairs,
        "policies": summaries, "configs": configs,
        "baseline_e4_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
    }
    (destination / "summary.json").write_bytes((json.dumps(summary, ensure_ascii=False, indent=2,
                                                          allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({key: value for key, value in summary.items() if key != "configs"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repair-run", type=Path)
    args = parser.parse_args()
    audit(args.run, args.baseline, args.destination, args.repair_run)

"""Select measured throughput leaders from comparable, complete fixed-question trials."""

import argparse
import json
from pathlib import Path


TESTS = {"E1": (96, (16, 32, 64, 96)), "E2": (48, (24, 48)),
         "E3": (96, (32, 64, 96)), "E4": (48, (24, 48))}


def read(folder, file):
    return json.loads((folder / file).read_text(encoding="utf-8"))


def paired(first, second):
    left = {row["problem_id"]: row for row in read(first, "predictions.json")}
    right = {row["problem_id"]: row for row in read(second, "predictions.json")}
    assert left.keys() == right.keys()
    return {
        "questions": len(left),
        "different_verdicts": sum(left[i]["is_correct"] != right[i]["is_correct"] for i in left),
        "different_answers": sum(left[i]["predicted_answer"] != right[i]["predicted_answer"] for i in left),
    }


def choose(root, *, include_e4=True):
    cases = {}
    for case, (sample_count, workers) in TESTS.items():
        if case == "E4" and not include_e4:
            continue
        folders = {count: root / f"vllm-{case}-c{count}" for count in workers}
        summaries = {count: read(folder, "summary.json") for count, folder in folders.items()}
        ids = [read(folder, "config.json")["sample_ids"] for folder in folders.values()]
        assert all(group == ids[0] for group in ids)
        assert len(set(ids[0])) == sample_count
        for count, summary in summaries.items():
            assert summary["workers"] == count and summary["samples"] == sample_count and not summary["failed"]
            assert summary["native_files_verified_at_commit"] and summary["pinned_dataset_checked"]
            assert summary["case"] == case and summary["classification"] == "published_code"
        selected = max(workers, key=lambda count: summaries[count]["questions_per_minute"])
        weights = {}
        if case in ("E3", "E4"):
            for folder in folders.values():
                for row in read(folder, "predictions.json"):
                    index = row["problem_id"]
                    weights[index] = max(weights.get(index, 0), row["output_tokens"])
        priorities = sorted(weights, key=lambda index: (-weights[index], index))[:16]
        cases[case] = {
            "workers": selected, "sample_size": sample_count, "priority_question_ids": priorities,
            "trials": [summaries[count] for count in workers],
            "paired_vs_selected": {count: paired(folder, folders[selected]) for count, folder in folders.items()},
        }

    backend = {}
    for case, count in (("E1", 8), ("E3", 4)):
        native_dir, vllm_dir = (root / f"native-{case}-n{count}", root / f"vllm-{case}-match-native")
        native, vllm = (read(native_dir, "summary.json"), read(vllm_dir, "summary.json"))
        assert native["samples"] == vllm["samples"] == count and not native["failed"] and not vllm["failed"]
        backend[case] = {
            "native": native, "vllm": vllm, "measurement": paired(native_dir, vllm_dir),
            "speedup": native["seconds"] / vllm["seconds"],
        }
    return {
        "criterion": "Highest completed questions per minute on identical question IDs per case",
        "engine": {"version": "0.33.0", "prefix_cache": True, "max_sequences": 112, "batched_tokens": 8192,
                   "gpu_memory_utilization": 0.60, "max_model_len": 32768, "dtype": "bfloat16"},
        "backend_benchmark": backend, "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--without-e4", action="store_true")
    args = parser.parse_args()
    profile = choose(args.root, include_e4=not args.without_e4)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for case, selected in profile["cases"].items():
        print(case, selected["workers"], max(trial["questions_per_minute"] for trial in selected["trials"]))

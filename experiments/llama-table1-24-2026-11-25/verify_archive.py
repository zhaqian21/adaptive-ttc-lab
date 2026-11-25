"""Verify the archived Llama predictions and print the paper Table 1 comparison."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPSTREAM = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"
MODEL = ["model", "meta-llama/Llama-3.1-8B-Instruct", "359efdbb8af05b788a4ad4185215c6b8caa9052c"]
CASES = {"E1": 1, "E2": 10, "E3": 1, "E4": 10}
BENCHMARKS = {
    "math": ("MATH-500", 500, (219, 223, 236, 327), ("43.8%", "44.6%", "47.2%", "65.4%")),
    "aime24": ("AIME24", 30, (1, 2, 2, 3), ("3.3%", "6.67%", "6.67%", "10.0%")),
    "amo": ("AMO-Bench", 50, (1, 1, 2, 2), ("2.0%", "2.0%", "4.0%", "4.0%")),
}


def read(path):
    return json.loads(path.read_bytes())


def require(condition, label):
    if not condition:
        raise ValueError(label)


def verify_case(root, dataset, case, size, paper_correct, dataset_sha256, exported):
    folder = root / dataset / case
    audit = read(folder / "audit.json")
    payload = (folder / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    iterations = CASES[case]
    correct = sum(row["is_correct"] is True for row in rows)
    require((audit["dataset"], audit["case"], audit["upstream_commit"], audit["vllm"])
            == (dataset, case, UPSTREAM, "0.33.0"), f"{folder}: evaluation metadata")
    require(audit["predictions_sha256"] == hashlib.sha256(payload).hexdigest(),
            f"{folder}: predictions SHA-256")
    require((audit["samples"], audit["correct"], audit["paper_correct"], audit["accuracy"])
            == (size, correct, paper_correct, correct / size), f"{folder}: accuracy or paper count")
    require(exported == {key: audit[key] for key in exported}, f"{folder}: final coordinator summary")
    ids = [row["dataset_unique_id"] for row in rows]
    require([row["problem_id"] for row in rows] == list(range(size)) and len(set(ids)) == size,
            f"{folder}: question coverage")
    require(all(row["lane"] == row["problem_id"] % 2 and
                1 <= row["selected_iteration"] <= iterations and
                type(row["is_correct"]) in (bool, type(None)) and
                (row["is_correct"] is not True or row["candidate_oracle_correct"] is True)
                for row in rows), f"{folder}: shard assignment, selection or verdict")
    require(len(audit["configs"]) == len(audit["shards"]) == 2, f"{folder}: shard count")
    for lane, (config, shard) in enumerate(zip(audit["configs"], audit["shards"])):
        lane_rows = rows[lane::2]
        expected = list(range(lane, size, 2))
        eval_config = config["eval"]
        require(config["sample_ids"] == expected and sorted(config["submission_order"]) == expected,
                f"{folder}: planned sample IDs on GPU {lane}")
        require(config["seed"] == 0 and config["dataset_sha256"] == dataset_sha256 and
                MODEL in config["datasets_and_models"] and
                eval_config["num_iterations"] == iterations and
                eval_config["use_prm_selection"] == (iterations > 1),
                f"{folder}: model, seed or iteration count on GPU {lane}")
        require((shard["samples"], shard["correct"], shard["generation_calls"], shard["output_tokens"])
                == (len(lane_rows), sum(row["is_correct"] is True for row in lane_rows),
                    sum(row["llm_calls"] for row in lane_rows), sum(row["output_tokens"] for row in lane_rows)),
                f"{folder}: per-shard totals on GPU {lane}")
        require(shard["prm_calls"] == (iterations * len(lane_rows) if iterations > 1 else 0),
                f"{folder}: PRM request count on GPU {lane}")
    return correct, ids


def markdown_table(root):
    summary = read(root / "summary.json")
    record = read(root.parent / "run_record.json")
    require(summary["pipeline_sha256"] == record["source_sha256"]["experiments/llama-table1-825/llama_table1.py"],
            "Coordinator source fingerprint")
    for file, expected in record["source_sha256"].items():
        source = (root.parent.parent.parent / file).read_bytes().replace(b"\r\n", b"\n")
        require(hashlib.sha256(source).hexdigest() == expected, f"{file}: source fingerprint")
    require(summary["model"] == MODEL and set(summary["cases"]) == set(BENCHMARKS),
            "Final summary model or benchmark coverage")
    lines = [
        "| 数据集 | 设置 | 公开代码复现 | 论文表 1 | 差值（百分点） |",
        "|---|---|---:|---:|---:|",
    ]
    total = 0
    for dataset, (name, size, counts, published) in BENCHMARKS.items():
        cases = summary["cases"][dataset]
        require(set(cases) == set(CASES), f"{dataset}: missing or unexpected settings")
        baseline_ids = None
        for (case, paper_correct, paper_accuracy) in zip(CASES, counts, published):
            correct, ids = verify_case(root, dataset, case, size, paper_correct,
                                       record["datasets"][dataset]["sha256"], cases[case])
            if baseline_ids is None:
                baseline_ids = ids
            require(ids == baseline_ids, f"{dataset}/{case}: dataset IDs differ from E1")
            accuracy = f"{correct / size:.1%}"
            delta = 100 * (correct - paper_correct) / size
            delta_text = f"{delta:+.2f}".rstrip("0").rstrip(".") if delta else "0"
            lines.append(f"| {name} | {case} | {correct}/{size}（{accuracy}） | {paper_accuracy} | {delta_text} |")
            total += size
    require(total == summary["total_predictions"] == 2320, "Total prediction count")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table(HERE / "results"))

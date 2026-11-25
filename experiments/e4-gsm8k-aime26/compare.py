"""Audit the candidate pool and replay the five PR #31 policies."""
import argparse
import json
from pathlib import Path
import sys

from benchmarks import DATASETS, sha


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e4-small-dynamic-k"))
from dynamic_k import MODES, replay, summarize
from load_pools import K, PREFIX, SMALL_K, make_row, tokens
from compare_online import verify_row
from replay_selection import encode


MODELS = ("llama", "qwen")
DATASET_LABELS = {"gsm8k": "GSM8K", "aime26": "AIME 2026"}
MODEL_LABELS = {"llama": "Llama", "qwen": "Qwen"}
METHOD_LABELS = dict(zip(MODES, ("固定 3 条", "固定 5 条", "固定 10 条（E4）",
                                 "五条一致早停", "动态 3/5")))


def read(path):
    return json.loads(path.read_bytes())


def save_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def write_rows(path, rows):
    payload = b"".join(encode(row) for row in rows)
    path.write_bytes(payload)
    return sha(payload)


def export(run, output, questions):
    generation = read(run / "generation.json")
    if generation["worker_exit_codes"] != dict.fromkeys(MODELS, 0):
        raise ValueError("Both model workers must have completed")
    for lane, model in enumerate(MODELS):
        source = run / model
        expected = {source / "problems" / f"{i:03d}.json" for i in range(len(questions))}
        if set((source / "problems").glob("*.json")) != expected:
            raise ValueError(f"{model}: incomplete or extra source rows")
        config = read(source / "config.json")
        if (config["lane"] != lane or config["num_iterations"] != K
                or config["manifest_sha256"] != sha((HERE / "questions.json").read_bytes())
                or config["source_sha256"] != sha((HERE / "run.py").read_bytes())
                or config["online_ab_sha256"] != sha((HERE.parent / "llama-e4-online-ab/online_ab.py").read_bytes())):
            raise ValueError(f"{model}: generation provenance mismatch")
        save_json(output / f"{model}_config.json", config)
        for dataset in DATASETS:
            paths = [source / "problems" / f"{q['problem_id']:03d}.json" for q in questions if q["dataset"] == dataset]
            write_rows(output / f"{model}_{dataset}_pool.jsonl", [read(path) for path in paths])
    save_json(output / "generation.json", generation)


def compare(pool, questions, lane):
    if len(pool) != len(questions):
        raise ValueError("Unexpected number of candidates pools")
    rows = []
    for raw, question in zip(pool, questions):
        ident = ("problem_id", "dataset_unique_id", "dataset")
        if (any(raw[key] != question[key] for key in ident) or
                raw["gold_answer"] != question["gold_answer_extracted"]):
            raise ValueError("Pool and question manifest differ")
        verify_row(raw, "fixed", expected_lane=lane)
        costs = raw["attempts"][-1]["stages"][:-1]
        paired = replay(make_row(raw, raw["candidates"], raw["original_choice"]["selected_iteration"],
                                 tokens(costs[:SMALL_K]), tokens(costs[:PREFIX]), tokens(costs)))
        paired["gold_answer"] = raw["gold_answer"]
        for mode in MODES:
            verdict = paired[mode]
            lost = paired["fixed10"]["is_correct"] and not verdict["is_correct"]
            verdict.update(lost_vs_fixed10=lost, unanimous_stop=verdict["prm_requests"] == 0,
                           lost_on_unanimous_stop_vs_fixed10=lost and verdict["prm_requests"] == 0)
        dynamic = paired["dynamic3_to_5"]
        dynamic["lost_due_to_stop_vs_fixed5"] = (dynamic["unanimous_stop"] and paired["fixed5"]["is_correct"]
                                                 and not dynamic["is_correct"])
        rows.append(paired)

    result = summarize(rows)
    for mode, stats in result["methods"].items():
        stats["accuracy"] = stats["correct"] / len(rows)
        stats["mean_candidates"] = stats["candidates"] / len(rows)
        for key in ("lost_vs_fixed10", "unanimous_stop", "lost_on_unanimous_stop_vs_fixed10"):
            stats[key] = sum(row[mode][key] for row in rows)
    result["dynamic_early_stop_losses_vs_fixed5"] = sum(
        row["dynamic3_to_5"]["lost_due_to_stop_vs_fixed5"] for row in rows
    )
    result["retried_questions"] = sum(len(row["attempts"]) > 1 for row in pool)
    result["discarded_attempts"] = sum(len(row["attempts"]) - 1 for row in pool)
    result["retained_generation_without_usage"] = sum(
        stage["generation_without_usage"] for row in pool for stage in row["attempts"][-1]["stages"]
    )
    result["discarded_generation_without_usage"] = sum(
        stage["generation_without_usage"] for row in pool for attempt in row["attempts"][:-1] for stage in attempt["stages"]
    )
    return rows, result


def table(summary):
    lines = [
        "| 数据集 | 模型 | 规则 | 正确率 | 平均候选数 | 生成 token | 外层 PRM 请求 | 固定 10 对、该规则错 | 一致早停且固定 10 对、早停错 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model in MODELS:
            group = summary[f"{model}_{dataset}"]
            for mode, stats in group["methods"].items():
                lost = stats["lost_on_unanimous_stop_vs_fixed10"] if mode in ("early5_to_10", "dynamic3_to_5") else "—"
                lines.append(
                    f"| {DATASET_LABELS[dataset]} | {MODEL_LABELS[model]} | {METHOD_LABELS[mode]} | "
                    f"{stats['accuracy']:.1%} ({stats['correct']}/{group['samples']}) | {stats['mean_candidates']:.2f} | "
                    f"{stats['tokens']:,} | {stats['prm_requests']:,} | {stats['lost_vs_fixed10']} | {lost} |"
                )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, help="Completed generation results, required for first export")
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read(HERE / "questions.json")
    questions = plan["questions"]
    if (plan["datasets"] != DATASETS or [q["problem_id"] for q in questions] != list(range(len(questions)))
            or len({(q["dataset"], q["dataset_unique_id"]) for q in questions}) != len(questions)):
        raise ValueError("Invalid question manifest")
    if args.run_root:
        args.pools.mkdir(parents=True, exist_ok=True)
        export(args.run_root, args.pools, questions)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for lane, model in enumerate(MODELS):
        for dataset, spec in DATASETS.items():
            group = [q for q in questions if q["dataset"] == dataset]
            if len(group) != spec["samples"]:
                raise ValueError("Manifest sample count mismatch")
            name = f"{model}_{dataset}"
            payload = (args.pools / f"{name}_pool.jsonl").read_bytes()
            rows, stats = compare([json.loads(line) for line in payload.splitlines()], group, lane)
            summary[name] = {**stats, "pool_sha256": sha(payload),
                             "paired_sha256": write_rows(args.output / f"{name}_paired.jsonl", rows)}
    save_json(args.output / "summary.json", summary)
    (args.output / "table.md").write_bytes(table(summary).encode("utf-8"))
    print(table(summary), end="")


if __name__ == "__main__":
    main()

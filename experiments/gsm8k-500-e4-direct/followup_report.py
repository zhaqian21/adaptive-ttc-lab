"""Verify the five benchmark direct-only results and summarize them with GSM8K."""
import argparse
import json
from pathlib import Path

from sample import HERE, sha
from compare import read, save_json, write_rows


PLAN = HERE / "followup_questions.json"
LANES = {"qwen": (0,), "llama": (1, 7)}
LABELS = {"gsm8k": "GSM8K", "math": "MATH-500", "amc23": "AMC23", "aime24": "AIME24",
          "amo": "AMO-Bench", "aime26": "AIME 2026"}


def collect(run, output, questions):
    runtime = read(run / "generation.json")
    if runtime["exit_codes"] != {"0": 0, "1": 0, "7": 0}:
        raise ValueError("All three workers must have exited successfully")
    save_json(output / "generation.json", runtime)
    for model, lanes in LANES.items():
        rows = []
        for lane in lanes:
            path = run / f"gpu-{lane}"
            settings = read(path / "config.json")
            expected = [q for q in questions if lane == 0 or q["problem_id"] % 2 == (lane == 7)]
            ids = [q["problem_id"] for q in expected]
            if (settings["lane"] != lane or settings["ids"] != ids
                    or settings["plan_sha256"] != sha(PLAN.read_bytes())
                    or settings["source_sha256"] != sha((HERE / "followup.py").read_bytes())
                    or settings["shared_direct_source_sha256"] != sha((HERE / "evaluate.py").read_bytes())):
                raise ValueError(f"GPU {lane}: unexpected generation provenance")
            expected_paths = [path / "direct" / f"{i:03d}.json" for i in ids]
            if set((path / "direct").glob("*.json")) != set(expected_paths):
                raise ValueError(f"GPU {lane}: incomplete or extra predictions")
            rows.extend(read(q) for q in expected_paths)
            save_json(output / f"gpu-{lane}_config.json", settings)
        write_rows(output / f"{model}_direct.jsonl", sorted(rows, key=lambda r: r["problem_id"]))


def summarize(archive, questions, sources):
    output = {name: {"samples": spec["samples"]} for name, spec in sources.items()}
    for model in LANES:
        path = archive / f"{model}_direct.jsonl"
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        if len(rows) != len(questions):
            raise ValueError(f"{model}: incomplete sample")
        for row, question in zip(rows, questions):
            index = question["problem_id"]
            lane = 0 if model == "qwen" else (7 if index % 2 else 1)
            if (any(row[k] != question[k] for k in ("problem_id", "source_problem_id", "dataset", "dataset_unique_id"))
                    or row["gold_answer"] != question["gold_answer_extracted"] or row["seed"] != index
                    or type(row["is_correct"]) is not bool or row["lane"] != lane or row["prm_requests"] != 0
                    or row["generation_requests"] - row["generation_without_usage"] != 1):
                raise ValueError(f"Invalid paired generation record for {model}/{index}")
        for dataset, stats in output.items():
            group = [r for r in rows if r["dataset"] == dataset]
            if len(group) != stats["samples"]:
                raise ValueError(f"Unexpected row count for {dataset}")
            correct = sum(r["is_correct"] for r in group)
            stats[model] = {
                "correct": correct, "accuracy": correct / len(group),
                "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in group), "prm_requests": 0,
                "generation_without_usage": sum(r["generation_without_usage"] for r in group),
                "predictions_sha256": sha(payload),
            }
    return output


def outcome(stats, model):
    record = stats[model]
    return f"{record['accuracy']:.1%} ({record['correct']}/{stats['samples']})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gsm8k-summary", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = read(PLAN)
    questions, sources = [experiment[k] for k in ("questions", "sources")]
    if args.run:
        args.archive.mkdir(parents=True, exist_ok=True)
        collect(args.run, args.archive, questions)
    summary = summarize(args.archive, questions, sources)
    gsm8k = read(args.gsm8k_summary)
    summary = {"gsm8k": {"samples": gsm8k["llama"]["samples"],
                        **{m: gsm8k[m]["methods"]["direct"] for m in LANES}}, **summary}
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "summary.json", summary)
    lines = ["| 数据集 | 题数 | Qwen 单次直接作答 | Llama 单次直接作答 |",
             "|---|---:|---:|---:|"]
    for dataset, stats in summary.items():
        lines.append(f"| {LABELS[dataset]} | {stats['samples']} | {outcome(stats, 'qwen')} | {outcome(stats, 'llama')} |")
    text = "\n".join(lines) + "\n"
    (args.output / "table.md").write_bytes(text.encode("utf-8"))
    print(text, end="")

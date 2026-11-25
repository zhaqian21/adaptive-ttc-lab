"""Archive the three lanes and compare direct generation with five paired E4 rules."""
import argparse
import json
from pathlib import Path

from sample import HERE, PREVIOUS, REUSED, TOTAL, sha
from compare import METHOD_LABELS, MODEL_LABELS, compare, read, save_json, summarize, write_rows


LANES = {"qwen": (0,), "llama": (1, 7)}
PREVIOUS_LANES = {"llama": 0, "qwen": 1}
PR32 = "ca0d95036a073ba22a2eb5c53d0a14bf464b2fb0"


def lane_ids(lane, start):
    return [i for i in range(start, TOTAL) if lane == 0 or i % 2 == (lane == 7)]


def jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def collect(run, pools, prior, plan):
    runtime = read(run / "generation.json")
    if runtime["worker_exit_codes"] != {"0": 0, "1": 0, "7": 0}:
        raise ValueError("Expected all three successful workers")
    save_json(pools / "generation.json", runtime)
    for model, lanes in LANES.items():
        rows = {"direct": [], "e4": []}
        earlier = read(prior / f"{model}_config.json")
        for lane in lanes:
            root = run / f"gpu-{lane}"
            config = read(root / "config.json")
            shared = ("model", "prm", "decode", "engine", "agent", "upstream_commit")
            if any(config[key] != earlier[key] for key in shared):
                raise ValueError(f"GPU {lane}: historical and new E4 settings differ")
            if (config["manifest_sha256"] != sha((HERE / "questions.json").read_bytes())
                    or config["runner_sha256"] != sha((HERE / "evaluate.py").read_bytes())
                    or config["e4_solve_sha256"] != earlier["online_ab_sha256"]
                    or config["lane"] != lane):
                raise ValueError("Wrong generator provenance")
            for phase, first in (("direct", 0), ("e4", REUSED)):
                ids = lane_ids(lane, first)
                path = root / phase
                expected = [path / f"{i:03d}.json" for i in ids]
                if config[f"{phase}_ids"] != ids or set(path.glob("*.json")) != set(expected):
                    raise ValueError(f"GPU {lane}: incomplete or repeated {phase} checkpoints")
                rows[phase].extend(read(p) for p in expected)
            save_json(pools / f"gpu-{lane}_config.json", config)
        for phase, filename in (("direct", "direct"), ("e4", "e4_extra")):
            write_rows(pools / f"{model}_{filename}.jsonl", sorted(rows[phase], key=lambda q: q["problem_id"]))


def score_direct(rows, questions, model):
    if len(rows) != TOTAL:
        raise ValueError(f"{model}: incomplete direct predictions")
    for row, question in zip(rows, questions):
        index = question["problem_id"]
        lane = 0 if model == "qwen" else (7 if index % 2 else 1)
        if ((row["problem_id"], row["seed"], row["dataset_unique_id"], row["gold_answer"], row["lane"]) !=
                (index, index, question["dataset_unique_id"], question["gold_answer_extracted"], lane)
                or row["prm_requests"] != 0 or row["generation_requests"] - row["generation_without_usage"] != 1
                or type(row["is_correct"]) is not bool or min(row["input_tokens"], row["output_tokens"]) < 1):
            raise ValueError(f"{model}/{index}: invalid direct prediction")
    return {
        "correct": sum(r["is_correct"] for r in rows), "mean_candidates": 1.0,
        "prm_requests": 0, "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
        "generation_without_usage": sum(r["generation_without_usage"] for r in rows),
        "unanimous_stop": 0, "lost_on_unanimous_stop_vs_fixed10": 0,
    }


def complete(model, questions, pools, prior, output):
    frozen = prior / f"{model}_gsm8k_pool.jsonl"
    old_summary = read(prior / "summary.json")[f"{model}_gsm8k"]
    if sha(frozen.read_bytes()) != old_summary["pool_sha256"]:
        raise ValueError("Prior pool checksum mismatch")
    old = jsonl(frozen)
    direct_path = pools / f"{model}_direct.jsonl"
    extra_path = pools / f"{model}_e4_extra.jsonl"
    direct, extra = jsonl(direct_path), jsonl(extra_path)
    if [row["problem_id"] for row in extra] != list(range(REUSED, TOTAL)):
        raise ValueError("Additional E4 candidates must include each new question exactly once")
    rows, older = compare(old, questions[:REUSED], PREVIOUS_LANES[model])
    batches = [older]
    for lane in LANES[model]:
        ids = lane_ids(lane, REUSED)
        pairs, details = compare([extra[i - REUSED] for i in ids], [questions[i] for i in ids], lane)
        rows.extend(pairs)
        batches.append(details)
    rows.sort(key=lambda r: r["problem_id"])
    if [r["problem_id"] for r in rows] != list(range(TOTAL)):
        raise ValueError("The merged E4 rows are incomplete")
    report = summarize(rows)
    baseline = score_direct(direct, questions, model)
    baseline["lost_vs_fixed10"] = sum(r["fixed10"]["is_correct"] and not d["is_correct"] for r, d in zip(rows, direct))
    report["methods"] = {"direct": baseline, **report["methods"]}
    for mode, stats in report["methods"].items():
        stats["accuracy"] = stats["correct"] / TOTAL
        if mode == "direct":
            continue
        stats["mean_candidates"] = stats["candidates"] / TOTAL
        for key in ("lost_vs_fixed10", "unanimous_stop", "lost_on_unanimous_stop_vs_fixed10"):
            stats[key] = sum(r[mode][key] for r in rows)
    for key in ("retried_questions", "retained_generation_without_usage", "dynamic_early_stop_losses_vs_fixed5"):
        report[key] = sum(batch[key] for batch in batches)
    for row, d in zip(rows, direct):
        row["direct"] = {
            "predicted_answer": d["predicted_answer"], "is_correct": d["is_correct"], "k": 1,
            "tokens": d["input_tokens"] + d["output_tokens"], "prm_requests": 0,
            "lost_vs_fixed10": row["fixed10"]["is_correct"] and not d["is_correct"],
        }
    report["correct_by_subset"] = {
        mode: {"reused_200": sum(r[mode]["is_correct"] for r in rows[:REUSED]),
               "extra_300": sum(r[mode]["is_correct"] for r in rows[REUSED:])}
        for mode in report["methods"]
    }
    report["sources"] = {"pr32_commit": PR32, "pr32_pool_sha256": old_summary["pool_sha256"],
                         "direct_sha256": sha(direct_path.read_bytes()), "extra_pool_sha256": sha(extra_path.read_bytes()),
                         "paired_sha256": write_rows(output / f"{model}_paired.jsonl", rows)}
    return report


def table(summary):
    lines = [
        "| 模型 | 规则 | 正确率 | 平均候选数 | 生成 token | 外层 PRM 请求 | 固定 10 对、本规则错 | 其中一致早停丢题 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("llama", "qwen"):
        for mode, m in summary[model]["methods"].items():
            early_losses = m["lost_on_unanimous_stop_vs_fixed10"] if mode in ("early5_to_10", "dynamic3_to_5") else "—"
            lines.append(
                f"| {MODEL_LABELS[model]} | {METHOD_LABELS.get(mode, '单次直接作答')} | {m['accuracy']:.1%} ({m['correct']}/{TOTAL}) | "
                f"{m['mean_candidates']:.2f} | {m['tokens']:,} | {m['prm_requests']:,} | {m['lost_vs_fixed10']} | {early_losses} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--prior", type=Path, default=PREVIOUS / "results")
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read(HERE / "questions.json")["questions"]
    if ([q["problem_id"] for q in plan] != list(range(TOTAL))
            or len({q["dataset_unique_id"] for q in plan}) != TOTAL):
        raise ValueError("The 500-question plan must have unique identities")
    if args.run:
        args.pools.mkdir(parents=True, exist_ok=True)
        collect(args.run, args.pools, args.prior, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {model: complete(model, plan, args.pools, args.prior, args.output) for model in LANES}
    save_json(args.output / "summary.json", summary)
    (args.output / "table.md").write_bytes(table(summary).encode("utf-8"))
    print(table(summary), end="")

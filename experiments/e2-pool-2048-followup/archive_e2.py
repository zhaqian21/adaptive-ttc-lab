"""Verify and package the completed E2 cohorts and schedule decisions."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

from schedule import COHORTS, COUNTS, HERE


ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "direct-e4-pools-2048"))
from archive_pools import digest, e4 as replay_pool, encode, read, save


MANIFEST = ROOT / "gsm8k-500-e4-direct/followup_questions.json"
LANES = {"qwen": (0,), "llama": (1, 7)}
CHUNK = 25


def plan():
    frozen = read(MANIFEST)
    return {ds: [{**q, "problem_id": q["source_problem_id"]} for q in frozen["questions"] if q["dataset"] == ds]
            for ds in COHORTS}


def validate(q, row, lane, model):
    i = q["problem_id"]
    if (any(q[key] != row[key] for key in ("problem_id", "dataset", "dataset_unique_id"))
            or row["gold"] != q["gold_answer_extracted"] or row["lane"] != lane):
        raise ValueError(f"{model}/{q['dataset']}/{i}: pool identity differs")
    return replay_pool({"result": row}, {**q, "e4_seeds": {model: i}}, model)["fixed10"]


def export(run, out, planned):
    generation = read(run / "generation.json")
    schedule = generation["cohorts"]
    if list(schedule) != list(COHORTS) or schedule["math"]["status"] != "complete":
        raise ValueError("Main E2 cohort or schedule summary missing")
    save(out / "generation.json", generation)
    raw = out / "pools"
    raw.mkdir(exist_ok=True)
    for ds, entry in schedule.items():
        if entry["status"] == "deferred_by_window":
            continue
        if (entry["status"] != "complete" or entry["worker_exit_codes"] != {"0": 0, "1": 0, "7": 0}
                or len(planned[ds]) != COUNTS[ds]):
            raise ValueError(f"{ds}: incomplete generation")
        for model, lanes in LANES.items():
            rows = {}
            for lane in lanes:
                source = run / ds / f"gpu-{lane}"
                config = read(source / "config.json")
                ids = [q["problem_id"] for q in planned[ds] if lane == 0 or q["problem_id"] % 2 == (lane == 7)]
                if (config["case"] != "E2" or config["dataset"] != ds or config["ids"] != ids
                        or config["lane"] != lane or config["decode"]["max_tokens"] != 2048
                        or config["num_iterations"] != 10 or config["question_manifest_sha256"] != digest(MANIFEST.read_bytes())
                        or config["source_sha256"] != digest((HERE / "run_e2.py").read_bytes())
                        or config["solve_sha256"] != digest((ROOT / "llama-e4-online-ab/online_ab.py").read_bytes())):
                    raise ValueError(f"{ds}/GPU{lane}: unexpected generation settings")
                save(out / f"{ds}-gpu-{lane}_config.json", config)
                expected = {i: source / "problems" / f"{i:03d}.json" for i in ids}
                if set(expected.values()) != set((source / "problems").glob("*.json")):
                    raise ValueError(f"{ds}/GPU{lane}: incomplete or duplicate checkpoints")
                rows.update({i: read(path) for i, path in expected.items()})
            for c, start in enumerate(range(0, len(rows), CHUNK)):
                payload = b"".join(encode(rows[n]) for n in range(start, min(len(rows), start + CHUNK)))
                (raw / f"{model}_{ds}_{c:03d}.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))


def summary(archive, output, planned):
    generation = read(archive / "generation.json")
    results, paired = {}, []
    for dataset, run in generation["cohorts"].items():
        if run["status"] != "complete":
            continue
        results[dataset] = {}
        for model in LANES:
            chunks = sorted((archive / "pools").glob(f"{model}_{dataset}_*.jsonl.gz"))
            artifacts = {p.name: p.read_bytes() for p in chunks}
            rows = [json.loads(line) for buf in artifacts.values() for line in gzip.decompress(buf).splitlines()]
            if len(rows) != len(planned[dataset]):
                raise ValueError(f"{model}/{dataset}: incomplete archive")
            total = Counter()
            for q, row in zip(planned[dataset], rows):
                lane = 0 if model == "qwen" else (7 if q["problem_id"] % 2 else 1)
                verdict = validate(q, row, lane, model)
                total.update(correct=verdict["correct"], tokens=verdict["tokens"], prm_requests=verdict["prm_requests"],
                             retries=len(row["attempts"]) - 1)
                paired.append({"dataset": dataset, "model": model, "problem_id": q["problem_id"],
                               "dataset_unique_id": q["dataset_unique_id"], "gold": row["gold"], **verdict})
            results[dataset][model] = {"samples": len(rows), "accuracy": total["correct"]/len(rows), **total,
                                       "pools_sha256": {name: digest(buf) for name, buf in artifacts.items()}}
    payload = b"".join(encode(row) for row in paired)
    (output / "paired.jsonl").write_bytes(payload)
    save(output / "summary.json", {"generation": generation, "datasets": results, "paired_sha256": digest(payload)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    planned = plan()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run:
        args.archive.mkdir(parents=True, exist_ok=True)
        export(args.run, args.archive, planned)
    data = summary(args.archive, args.output, planned)
    labels = {"math":"MATH-500", "aime24":"AIME24", "amo":"AMO-Bench"}
    lines = ["| 数据集 | 模型 | E2 固定 10 条 | 生成 token | 外层 PRM 请求 |", "|---|---|---:|---:|---:|"]
    for ds, models in data.items():
        for model, row in models.items():
            lines.append(f"| {labels[ds]} | {model} | {row['correct']}/{row['samples']}（{row['accuracy']:.1%}） | "
                         f"{row['tokens']:,} | {row['prm_requests']:,} |")
    text = "\n".join(lines) + "\n"
    (args.output / "table.md").write_bytes(text.encode("utf-8"))
    print(text, end="")

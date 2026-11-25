"""Audit the 2,048-token candidate pools and derive paired results."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "math500-direct-continuation"))
from protocol import DRAWS, LONG, SHORT, continuation, key, majority
from comparison import DATASETS
from report import DATASET_LABELS, MODELS


PLAN = HERE / "question_plan.json"
LANES = {"qwen": (0,), "llama": (1, 7)}
DIRECT_METHODS = ("direct1", "vote3", "vote5")
E4_METHODS = ("fixed3", "fixed5", "fixed10", "early5_to_10", "dynamic3_to_5")
CHUNK = 25


def read(path):
    return json.loads(path.read_bytes())


def digest(buf):
    return hashlib.sha256(buf).hexdigest()


def encode(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def save(path, obj):
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def assigned(ids, lane):
    return [i for i in ids if lane == 0 or i % 2 == (lane == 7)]


def validate_identity(row, question, model):
    i = question["problem_id"]
    lane = 0 if model == "qwen" else (7 if i % 2 else 1)
    if (any(row[k] != question[k] for k in ("problem_id", "dataset", "dataset_unique_id"))
            or row["gold"] != question["gold_answer_extracted"] or row["lane"] != lane):
        raise ValueError(f"{model}/{i}: unexpected question or lane")


def validate_generation(phase, source, lane, ids):
    cfg = read(source / f"gpu-{lane}/config.json")
    if (cfg["lane"] != lane or cfg["phase"] != phase or cfg["decode"]["max_tokens"] != LONG
            or cfg["ids"] != assigned(ids, lane) or cfg["question_plan_sha256"] != digest(PLAN.read_bytes())
            or cfg["source_sha256"] != digest((HERE / "run_pools.py").read_bytes())
            or cfg["e4_source_sha256"] != digest((ROOT / "llama-e4-online-ab/online_ab.py").read_bytes())):
        raise ValueError(f"{phase}/GPU{lane}: generation profile differs")
    return cfg


def load_new_rows(source, planned, model):
    for q in planned:
        index = q["problem_id"]
        lane = LANES[model][0] if model == "qwen" or not index % 2 else 7
        path = source / f"gpu-{lane}/problems/{index:04d}.json"
        yield read(path)


def archive_new(source, output, phase, plan):
    ids = [q["problem_id"] for q in read(PLAN)]
    for lane in (0, 1, 7):
        save(output / f"gpu-{lane}_config.json", validate_generation(phase, source, lane, ids))
    if (source / "generation.json").is_file():
        generation = read(source / "generation.json")
        if generation["worker_exit_codes"] != {"0": 0, "1": 0, "7": 0}:
            raise ValueError("A worker failed")
        save(output / "generation.json", generation)
    archive = output / "pools"
    archive.mkdir(exist_ok=True)
    for dataset, planned in plan.items():
        for model in LANES:
            for chunk, start in enumerate(range(0, len(planned), CHUNK)):
                batch = planned[start:start + CHUNK]
                data = b"".join(encode(row) for row in load_new_rows(source, batch, model))
                name = f"{model}_{dataset}_{chunk:03d}.jsonl.gz"
                (archive / name).write_bytes(gzip.compress(data, mtime=0))


def direct(row, question):
    outputs = row["outputs"]
    if len(outputs) != DRAWS:
        raise ValueError("Direct requires exactly five candidates")
    for draw, raw in enumerate(outputs):
        n = raw["output_tokens"]
        if (raw["seed"] != question["direct_seed"] + 1_000_000 * draw or not 0 < n <= LONG
                or len(raw["token_ids"]) != n or raw["short"]["output_tokens"] != min(SHORT, n)
                or raw["full"]["output_tokens"] != n or raw["prm_requests"] != 0
                or raw["generation_requests"] - raw["generation_without_usage"] != 1
                or (raw["finish_reason"] == "length" and n != LONG)
                or (n <= SHORT and raw["full"] != raw["short"])):
            raise ValueError("Invalid Direct request or token ledger")
    pair = {}
    for cap, field in ((SHORT, "short"), (LONG, "full")):
        candidates = [raw[field] for raw in outputs]
        for name, n in zip(DIRECT_METHODS, (1, 3, 5)):
            pick = majority(candidates[:n])
            pair[f"{name}_{cap}"] = {
                **candidates[pick], "selected": pick + 1, "candidates": n, "prm_requests": 0,
                "tokens": sum(raw["input_tokens"] + c["output_tokens"]
                              for raw, c in zip(outputs[:n], candidates[:n])),
            }
    step = continuation(outputs[0])
    pair["direct1_on_demand"] = {
        **step["on_demand"], "candidates": 1, "selected": 1, "prm_requests": 0,
        "tokens": outputs[0]["input_tokens"] + step["on_demand"]["output_tokens"],
    }
    pair["extended"] = step["extended"]
    return pair


def e4(row, question, model):
    episode = row["result"]
    if (episode["problem_id"] != question["e4_seeds"][model] or episode["stopped_early"]
            or episode["dataset_unique_id"] != question["dataset_unique_id"]):
        raise ValueError("Wrong E4 identity or policy")
    stages = episode["attempts"][-1]["stages"]
    costs = [stage["input_tokens"] + stage["output_tokens"] for stage in stages[:-1]]
    cs = episode["candidates"]
    if ([c["iteration"] for c in cs] != list(range(1, 11))
            or [s["stage"] for s in stages] != [*range(1, 11), "prm"]
            or stages[-1]["prm_requests"] != 10 or any(v <= 0 for v in costs)
            or any(s["prm_requests"] for s in stages[:-1])):
        raise ValueError("Unexpected E4 stages")
    for c in cs:
        if (digest(c["answer_unstructured"].encode("utf-8")) != c["completion_sha256"]
                or not c["prm_input"] or c["prm_score"]["mean_reward"] != c["mean_reward"]
                or not 0 <= c["mean_reward"] <= 1):
            raise ValueError("Invalid retained trace or score")
    reward = [c["mean_reward"] for c in cs]
    labels = [key(c["predicted_answer"]) for c in cs]

    def pick(n):
        return max(range(n), key=reward.__getitem__) if any(reward[:n]) else n - 1

    def unanimous(n):
        return bool(labels[0]) and all(x == labels[0] for x in labels[:n])

    def choice(n, early=False):
        best = 0 if early else pick(n)
        return {"answer": cs[best]["predicted_answer"], "correct": cs[best]["is_correct"] is True,
                "selected": best + 1, "candidates": n, "prm_requests": 0 if early else n,
                "tokens": sum(costs[:n])}

    fixed = {f"fixed{n}": choice(n) for n in (3, 5, 10)}
    if fixed["fixed10"]["selected"] != episode["original_choice"]["selected_iteration"]:
        raise ValueError("Original E4 selection disagrees with the replay")
    fixed["early5_to_10"] = choice(5, early=True) if unanimous(5) else fixed["fixed10"]
    fixed["dynamic3_to_5"] = choice(3, early=True) if unanimous(3) else fixed["fixed5"]
    return fixed


def summary(phase, archive, out, plan):
    reports, pairs = {}, []
    for dataset, planned in plan.items():
        counts = reports[dataset] = {}
        for model in LANES:
            chunks = sorted((archive / "pools").glob(f"{model}_{dataset}_*.jsonl.gz"))
            rows, fingerprint = [], {}
            for path in chunks:
                buf = path.read_bytes()
                fingerprint[path.name] = digest(buf)
                rows.extend(json.loads(line) for line in gzip.decompress(buf).splitlines())
            if len(rows) != len(planned):
                raise ValueError(f"{model}/{dataset}: incomplete pool")
            methods = {}
            for row, q in zip(rows, planned):
                validate_identity(row, q, model)
                result = direct(row, q) if phase == "direct5" else e4(row, q, model)
                pair = {**{k: row[k] for k in ("problem_id", "dataset", "dataset_unique_id", "lane", "gold")},
                        "model": model, **result}
                pairs.append(pair)
                for name, val in result.items():
                    if isinstance(val, dict):
                        totals = methods.setdefault(name, Counter())
                        totals.update(correct=val["correct"], tokens=val["tokens"], prm_requests=val["prm_requests"],
                                      candidates=val["candidates"])
            counts[model] = {"samples": len(rows), "methods": methods, "pools_sha256": fingerprint}
    payload = b"".join(encode(row) for row in pairs)
    (out / "paired.jsonl").write_bytes(payload)
    save(out / "summary.json", {"phase": phase, "datasets": reports, "paired_sha256": digest(payload)})
    return reports


def table(phase, summary):
    columns = ("direct1_1024", "direct1_2048", "vote3_2048", "vote5_2048") if phase == "direct5" else E4_METHODS
    lines = ["| 数据集 | 模型 | " + " | ".join(columns) + " |",
             "|---|---|" + "---:|" * len(columns)]
    for dataset, peers in summary.items():
        for model, report in peers.items():
            n = report["samples"]
            def cell(name):
                m = report["methods"][name]
                return f"{m['correct']}/{n}（{m['correct']/n:.1%}） · {m['tokens']:,}"
            lines.append(f"| {DATASET_LABELS[dataset]} | {MODELS[model]} | " +
                         " | ".join(map(cell, columns)) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("direct5", "e4"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = read(PLAN)
    selected = (args.dataset,) if args.dataset else DATASETS
    plan = {d: [q for q in questions if q["dataset"] == d] for d in selected}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run:
        args.archive.mkdir(parents=True, exist_ok=True)
        archive_new(args.run / args.phase, args.archive, args.phase, plan)
    result = summary(args.phase, args.archive, args.output, plan)
    md = table(args.phase, result)
    (args.output / "table.md").write_bytes(md.encode("utf-8"))
    print(md, end="")

"""Audit frozen outputs and replay the token and voting comparisons."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from protocol import DRAWS, LONG, SHORT, STUDIES, continuation, key, majority


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "gsm8k-500-e4-direct"
LANES = {"qwen": (0,), "llama": (1, 7)}
MODELS = {"qwen": "Qwen", "llama": "Llama"}
LABELS = {"fixed1024": "固定 1,024", "fixed2048": "固定 2,048", "on_demand": "按需续写",
          "direct1": "单次作答", "vote3": "3 条投票", "vote5": "5 条投票"}


def read(path):
    return json.loads(path.read_bytes())


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def save(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def plans(study):
    qs = read(PRIOR / "followup_questions.json")["questions"][:500]
    if study == "voting":
        qs += [{**q, "problem_id": q["problem_id"] + 500} for q in read(PRIOR / "questions.json")["questions"]]
    if [q["problem_id"] for q in qs] != list(range(len(qs))):
        raise ValueError("Unexpected question plan")
    return qs


def collect(source, output, study, qs):
    run = source / study
    generation = read(run / "generation.json")
    if generation["worker_exit_codes"] != {"0": 0, "1": 0, "7": 0}:
        raise ValueError("Expected three successful workers")
    save(output / "generation.json", generation)
    for model, lanes in LANES.items():
        rows = []
        for lane in lanes:
            path = run / f"gpu-{lane}"
            config = read(path / "config.json")
            ids = [q["problem_id"] for q in qs if lane == 0 or q["problem_id"] % 2 == (lane == 7)]
            if (config["lane"] != lane or config["study"] != study or config["ids"] != ids
                    or config["source_sha256"] != digest((HERE / "run.py").read_bytes())
                    or config["decode"]["max_tokens"] != (LONG if study == "continuation" else SHORT)):
                raise ValueError(f"GPU {lane}: unexpected checkpoint provenance")
            files = [path / "problems" / f"{i:04d}.json" for i in ids]
            if set(files) != set((path / "problems").glob("*.json")):
                raise ValueError("Missing or extra question checkpoints")
            rows.extend(read(f) for f in files)
            save(output / f"gpu-{lane}_config.json", config)
        payload = b"".join(encode(row) for row in sorted(rows, key=lambda r: r["problem_id"]))
        (output / f"{model}_outputs.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))


def validate(row, question, study, model):
    index = question["problem_id"]
    lane = 0 if model == "qwen" else (7 if index % 2 else 1)
    if (any(row[k] != question[k] for k in ("problem_id", "dataset", "dataset_unique_id"))
            or row["gold"] != question["gold_answer_extracted"] or row["lane"] != lane):
        raise ValueError(f"Changed sample identity {model}/{index}")
    cap = LONG if study == "continuation" else SHORT
    if len(row["outputs"]) != (1 if study == "continuation" else DRAWS):
        raise ValueError("Wrong pool size")
    for draw, output in enumerate(row["outputs"]):
        n = output["output_tokens"]
        if (output["seed"] != index % 500 + 1_000_000 * draw or not 0 < n <= cap
                or output["finish_reason"] not in ("stop", "length")
                or (output["finish_reason"] == "length" and n != cap) or output["input_tokens"] < 1
                or output["prm_requests"] != 0 or output["generation_requests"] - output["generation_without_usage"] != 1):
            raise ValueError(f"Invalid request ledger for {model}/{index}/{draw}")
        if study == "continuation":
            if (len(output["token_ids"]) != n or output["short"]["output_tokens"] != min(SHORT, n)
                    or output["full"]["output_tokens"] != n
                    or (n <= SHORT and output["short"] != output["full"])):
                raise ValueError("Incorrect token prefix")
        elif digest(output["completion"].encode("utf-8")) != output["completion_sha256"]:
            raise ValueError("Changed archived completion")


def cost(verdict, input_tokens):
    return {**verdict, "tokens": input_tokens + verdict["output_tokens"], "prm_requests": 0}


def paired(row, study, legacy):
    pair = {k: row[k] for k in ("problem_id", "dataset", "dataset_unique_id", "gold")}
    outputs = row["outputs"]
    if study == "continuation":
        raw = outputs[0]
        methods = continuation(raw)
        extended = methods.pop("extended")
        pair.update({name: cost(verdict, raw["input_tokens"]) for name, verdict in methods.items()})
        pair["extended"] = extended
        first = pair["fixed1024"]
    else:
        for name, n in (("direct1", 1), ("vote3", 3), ("vote5", 5)):
            pick = 0 if n == 1 else majority(outputs[:n])
            chosen = outputs[pick]
            pair[name] = {k: chosen[k] for k in ("answer", "correct", "completion_sha256")}
            pair[name].update(selected_draw=pick + 1, candidates=n, prm_requests=0,
                              tokens=sum(c["input_tokens"] + c["output_tokens"] for c in outputs[:n]))
        first = pair["direct1"]
    pair["prior_direct"] = {"answer": legacy["predicted_answer"], "correct": legacy["is_correct"],
                            "same_completion": first["completion_sha256"] == legacy["completion_sha256"]}
    return pair


def summary(rows, names, study):
    report = {"samples": len(rows), "prior_direct_correct": sum(r["prior_direct"]["correct"] for r in rows),
              "prior_direct_same_completion": sum(r["prior_direct"]["same_completion"] for r in rows), "methods": {}}
    base = names[0]
    for method in names:
        values = [r[method] for r in rows]
        correct = sum(v["correct"] for v in values)
        report["methods"][method] = {
            "correct": correct, "accuracy": correct / len(rows), "tokens": sum(v["tokens"] for v in values),
            "prm_requests": sum(v["prm_requests"] for v in values),
            "wrong_to_right": sum(not r[base]["correct"] and r[method]["correct"] for r in rows),
            "right_to_wrong": sum(r[base]["correct"] and not r[method]["correct"] for r in rows),
        }
    if study == "continuation":
        report["at_1024_without_answer"] = sum(v["fixed1024"]["output_tokens"] == SHORT
                                               and not key(v["fixed1024"]["answer"]) for v in rows)
        report["extended"] = sum(r["extended"] for r in rows)
    return report


def reduce(study, archive, output, qs, prior_math, prior_gsm):
    names = ("fixed1024", "fixed2048", "on_demand") if study == "continuation" else ("direct1", "vote3", "vote5")
    datasets = ("math",) if study == "continuation" else ("math", "gsm8k")
    reports = {}
    for model in LANES:
        records = jsonl(prior_math / f"{model}_direct.jsonl") + jsonl(prior_gsm / f"{model}_direct.jsonl")
        legacy = {(r["dataset"], r["dataset_unique_id"]): r for r in records}
        path = archive / f"{model}_outputs.jsonl.gz"
        artifact = path.read_bytes()
        rows = [json.loads(line) for line in gzip.decompress(artifact).splitlines()]
        if len(rows) != len(qs):
            raise ValueError("Incomplete candidate archive")
        pairs = []
        for row, question in zip(rows, qs):
            validate(row, question, study, model)
            previous = legacy[(row["dataset"], row["dataset_unique_id"])]
            if previous["gold_answer"] != row["gold"]:
                raise ValueError("Previous Direct reference differs")
            pairs.append(paired(row, study, previous))
        content = b"".join(encode(row) for row in pairs)
        (output / f"{model}_paired.jsonl").write_bytes(content)
        reports[model] = {
            "outputs_sha256": digest(artifact), "paired_sha256": digest(content),
            "generation_without_usage": sum(c["generation_without_usage"] for r in rows for c in r["outputs"]),
            "datasets": {ds: summary([r for r in pairs if r["dataset"] == ds], names, study) for ds in datasets},
        }
    return reports


def table(reports):
    rows = ["| 数据集 | 模型 | 规则 | 正确率 | 生成 token | PRM 请求 | 错→对 | 对→错 |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for model, result in reports.items():
        for ds, dataset in result["datasets"].items():
            for mode, stats in dataset["methods"].items():
                label = "MATH-500" if ds == "math" else "GSM8K"
                rows.append(f"| {label} | {MODELS[model]} | {LABELS[mode]} | "
                            f"{stats['accuracy']:.1%} ({stats['correct']}/{dataset['samples']}) | {stats['tokens']:,} | "
                            f"{stats['prm_requests']:,} | {stats['wrong_to_right']} | {stats['right_to_wrong']} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-math", type=Path, default=PRIOR / "followup-results")
    parser.add_argument("--prior-gsm", type=Path, default=PRIOR / "results")
    args = parser.parse_args()
    qs = plans(args.study)
    if args.run:
        args.archive.mkdir(parents=True, exist_ok=True)
        collect(args.run, args.archive, args.study, qs)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = reduce(args.study, args.archive, args.output, qs, args.prior_math, args.prior_gsm)
    save(args.output / "summary.json", reports)
    text = table(reports)
    (args.output / "table.md").write_bytes(text.encode("utf-8"))
    print(text, end="")

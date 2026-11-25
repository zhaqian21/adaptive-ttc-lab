"""Compare the Direct voting cohorts with their original fixed-10 E4 archives."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from protocol import key
from report import DATASET_LABELS, HERE, LANES, MODELS, jsonl


DATASETS = ("gsm8k", "math", "amc23", "aime24", "aime26", "amo")
METHODS = ("direct1", "vote3", "vote5")


def e4_rows(model, dataset):
    experiments = HERE.parent
    if dataset == "gsm8k":
        path = experiments / "gsm8k-500-e4-direct/results" / f"{model}_paired.jsonl"
    elif dataset == "aime26":
        path = experiments / "e4-gsm8k-aime26/results" / f"{model}_aime26_paired.jsonl"
    else:
        path = experiments / "e4-small-dynamic-k/results" / f"{model}_{dataset}.jsonl"
    return jsonl(path)


def load_direct(model):
    by_dataset = defaultdict(list)
    for study in ("voting", "remaining"):
        archive = HERE / "results" / study
        pairs = jsonl(archive / f"{model}_paired.jsonl")
        raw = [json.loads(line) for line in gzip.decompress(
            (archive / f"{model}_outputs.jsonl.gz").read_bytes()).splitlines()]
        if len(raw) != len(pairs):
            raise ValueError(f"{model}/{study}: mismatched candidate and result counts")
        for record, pair in zip(raw, pairs):
            if any(record[k] != pair[k] for k in ("dataset", "dataset_unique_id", "problem_id")):
                raise ValueError("Different candidate and result identities")
            by_dataset[pair["dataset"]].append((pair, record["outputs"]))
    return by_dataset


def markdown():
    direct = {model: load_direct(model) for model in LANES}
    lines = [
        "| 数据集 | 模型 | Direct 1 条 | Direct 3 条 | Direct 5 条 | E4 固定 10 条 | 5 条含正确解 | 5 条全无可提取答案 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model, cohorts in direct.items():
            cases = cohorts[dataset]
            baseline = e4_rows(model, dataset)
            direct_ids = {r["dataset_unique_id"] for r, _ in cases}
            e4_ids = {r["dataset_unique_id"] for r in baseline}
            n = len(cases)
            if not n or n != len(baseline) or n != len(direct_ids) or direct_ids != e4_ids:
                raise ValueError(f"{model}/{dataset}: different Direct and E4 question sets")

            def cell(correct, tokens):
                return f"{correct}/{n}（{correct/n:.1%}） · {tokens:,}"

            direct_results = [cell(sum(r[method]["correct"] for r, _ in cases),
                                   sum(r[method]["tokens"] for r, _ in cases)) for method in METHODS]
            e4 = cell(sum(r["fixed10"]["is_correct"] for r in baseline),
                      sum(r["fixed10"]["tokens"] for r in baseline))
            oracle = sum(any(c["correct"] for c in candidates) for _, candidates in cases)
            missing = sum(not any(key(c["answer"]) for c in candidates) for _, candidates in cases)
            lines.append(f"| {DATASET_LABELS[dataset]} | {MODELS[model]} | " +
                         " | ".join([*direct_results, e4, str(oracle), str(missing)]) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_bytes(markdown().encode("utf-8"))

"""Freeze the exact six benchmark question identities and prior seed indices."""
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
BASELINE = EXPERIMENTS / "gsm8k-500-e4-direct"
sys.path.insert(0, str(EXPERIMENTS / "math500-direct-continuation"))
from comparison import DATASETS, e4_rows
from report import LANES


def read(path):
    return json.loads(path.read_bytes())


def freeze():
    prev = read(BASELINE / "followup_questions.json")["questions"]
    gs = read(BASELINE / "questions.json")["questions"]
    records = []
    for dataset in DATASETS:
        group = [q for q in (gs if dataset == "gsm8k" else prev) if q["dataset"] == dataset]
        ids = {q["dataset_unique_id"] for q in group}
        seeds = {}
        for model in LANES:
            baseline = e4_rows(model, dataset)
            seeds[model] = {q["dataset_unique_id"]: q["problem_id"] for q in baseline}
            if len(baseline) != len(group) or set(seeds[model]) != ids:
                raise ValueError(f"Different historical E4 population: {model}/{dataset}")
        for q in group:
            uid = q["dataset_unique_id"]
            records.append({**q, "problem_id": len(records), "direct_seed": q["problem_id"],
                            "e4_seeds": {m: s[uid] for m, s in seeds.items()}})
    if len(records) != 1150:
        raise ValueError("Expected 1,150 distinct questions")
    return records


if __name__ == "__main__":
    (HERE / "question_plan.json").write_bytes(
        (json.dumps(freeze(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

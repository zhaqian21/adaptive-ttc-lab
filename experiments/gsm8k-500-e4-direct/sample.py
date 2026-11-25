"""Extend the locked 200-question GSM8K sample to 500 without replacement."""
import argparse
import json
from pathlib import Path
from random import Random
import re
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "e4-gsm8k-aime26"
sys.path.insert(0, str(PREVIOUS))
from benchmarks import DATASETS, SAMPLE_SEED, manifest, sha


REUSED = 200
TOTAL = 500


def load(data):
    from pyarrow.parquet import read_table

    path = data / "gsm8k.parquet"
    spec = DATASETS["gsm8k"]
    if sha(path.read_bytes()) != spec["sha256"]:
        raise ValueError("GSM8K parquet checksum mismatch")
    rows = read_table(path).to_pylist()
    if len(rows) != spec["rows"]:
        raise ValueError("GSM8K test split has the wrong size")
    rng = Random(SAMPLE_SEED)
    first = sorted(rng.sample(range(len(rows)), REUSED))
    held = set(first)
    extra = sorted(rng.sample([i for i in range(len(rows)) if i not in held], TOTAL - REUSED))
    questions = []
    for i, original in enumerate(first + extra):
        row = rows[original]
        problem = row["question"]
        answer = row["answer"].rsplit("#### ", 1)[1].strip().replace(",", "")
        if not problem.strip() or not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", answer):
            raise ValueError(f"Invalid GSM8K source row {original}")
        questions.append({
            "problem_id": i, "dataset": "gsm8k", "dataset_unique_id": f"test:{original}",
            "level": None, "problem": problem, "gold_answer_extracted": answer,
        })
    if len({q["problem"] for q in questions}) != TOTAL:
        raise ValueError("Repeated GSM8K problem")
    return questions


def plan(questions):
    rows = manifest(questions)["questions"]
    original = json.loads((PREVIOUS / "questions.json").read_bytes())["questions"][:REUSED]
    if rows[:REUSED] != original:
        raise ValueError("Prior PR #32 question identities must stay unchanged")
    return {"dataset": {**DATASETS["gsm8k"], "samples": TOTAL}, "seed": SAMPLE_SEED,
            "previous_questions": REUSED, "additional_questions": TOTAL - REUSED, "questions": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    print(json.dumps(plan(load(args.data)), indent=2))

"""Load the frozen GSM8K sample and all AIME 2026 questions."""
import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import re


SAMPLE_SEED = 42
DATASETS = {
    "gsm8k": {
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "file": "main/test-00000-of-00001.parquet",
        "split": "test", "rows": 1319, "samples": 200,
        "sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    },
    "aime26": {
        "repo": "MathArena/aime_2026",
        "revision": "d61f486805306e6e6b5913504c07763a99e1b112",
        "file": "data/train-00000-of-00001.parquet",
        "split": "train", "rows": 30, "samples": 30,
        "sha256": "ae72853349a46c5f08666feb42db2d09fb4ac4fd4768c726b29702df21c7768e",
    },
}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def rows_by_dataset(questions):
    return {name: [q for q in questions if q["dataset"] == name] for name in DATASETS}


def load(data):
    from pyarrow.parquet import read_table

    questions = []
    for name, spec in DATASETS.items():
        path = data / (name + ".parquet")
        if sha(path.read_bytes()) != spec["sha256"]:
            raise ValueError(f"{name}: dataset hash mismatch")
        rows = read_table(path).to_pylist()
        if len(rows) != spec["rows"]:
            raise ValueError(f"{name}: unexpected source size")
        indices = list(range(len(rows)))
        if name == "gsm8k":
            indices = sorted(Random(SAMPLE_SEED).sample(indices, spec["samples"]))
        elif [row["problem_idx"] for row in rows] != list(range(1, 31)):
            raise ValueError("AIME 2026 problem indices must be 1 through 30")
        for index in indices:
            row = rows[index]
            if name == "gsm8k":
                problem = row["question"]
                answer = row["answer"].rsplit("#### ", 1)[1].strip().replace(",", "")
                valid = re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", answer)
                uid = f"test:{index}"
            else:
                problem = row["problem"]
                answer = str(row["answer"])
                valid = type(row["answer"]) is int and 0 <= row["answer"] <= 999
                uid = str(row["problem_idx"])
            if not problem.strip() or not valid:
                raise ValueError(f"{name}/{uid}: invalid source question or answer")
            questions.append({
                "problem_id": len(questions), "dataset": name, "dataset_unique_id": uid,
                "level": None, "problem": problem, "gold_answer_extracted": answer,
            })
    if len({(q["dataset"], q["problem"]) for q in questions}) != len(questions):
        raise ValueError("Duplicate source question")
    return questions


def manifest(questions):
    return {
        "datasets": DATASETS, "gsm8k_sample_seed": SAMPLE_SEED,
        "questions": [
            {**{key: q[key] for key in ("problem_id", "dataset", "dataset_unique_id", "gold_answer_extracted")},
             "question_sha256": sha(q["problem"].encode("utf-8"))}
            for q in questions
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    print(json.dumps(manifest(load(args.data)), indent=2))

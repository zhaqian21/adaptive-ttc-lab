"""Pinned datasets and model snapshots for the E4 transfer studies."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

from llama_table1 import MODEL as LLAMA


QWEN = ("model", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
AMC23 = {
    "repo": "math-ai/amc23",
    "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
    "split": "test",
    "sha256": "b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e",
    "samples": 40,
}
AMC23_PLAN = {
    "samples": 40, "smoke_samples": 8, "plan": "llama_amc23_plan.json",
    "history_sha256": "de1d0b05a7fc29f8b437468a257f684300a034643baeca7ccd302bbd26f5a1e5",
}
FOLLOWUPS = {
    "llama-amc23": {"model": LLAMA, **AMC23_PLAN},
    "qwen-amc23": {"model": QWEN, **AMC23_PLAN},
    "qwen-math500": {
        "model": QWEN, "samples": 500, "smoke_samples": 20,
        "plan": "qwen_math500_plan.json", "history_sha256": "328cb3679d83f7ccfee7e4f6f9a5a68fc1c46e5986eb798933aee66b62f3405f",
    },
}


def load_followup_benchmark(study, released):
    if study == "qwen-math500":
        return released.load_benchmark(SimpleNamespace(dataset="math"))
    if study not in ("llama-amc23", "qwen-amc23"):
        raise ValueError(f"Unsupported E4 transfer study: {study}")

    from datasets import Dataset

    path = released.runner.ROOT / "datasets/amc23.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != AMC23["sha256"]:
        raise ValueError("AMC23 parquet does not match the pinned digest")
    source = Dataset.from_parquet(str(path))
    if len(source) != AMC23["samples"] or len(set(source["id"])) != len(source):
        raise ValueError("Expected all 40 unique AMC23 questions")

    evaluator = released.runner.upstream.get_evaluator("MATH")
    questions = []
    for index, row in enumerate(source):
        answer = str(row["answer"]).strip()
        gold = evaluator._extract_answer_from_model_completion(r"\boxed{" + answer + "}")
        if not row["question"].strip() or not gold or not evaluator._check_answers_equiv(gold, answer):
            raise ValueError(f"AMC23 grading control failed at source ID {row['id']}")
        questions.append({
            "problem_id": index, "dataset_unique_id": str(row["id"]), "level": row.get("level"),
            "problem": row["question"], "gold_answer_extracted": gold,
        })
    return questions, {**AMC23, "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

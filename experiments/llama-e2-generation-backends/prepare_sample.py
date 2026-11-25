"""Freeze MATH-500 question IDs, prompts, and per-candidate seeds before generation."""

import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path


DATASET_SHA = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
UPSTREAM = "d9ca5f71f5c7d155153e8f7decd8b1c524968775"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "359efdbb8af05b788a4ad4185215c6b8caa9052c"
ASSET_NAME = f"meta-llama--Llama-3.1-8B-Instruct--{MODEL_REVISION}"
MODEL_CHOICES = 10
SELECTION_KEY = "llama-e2-generation-backends-v1/2026-11-25/"
PRM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def encode(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def freeze_sample(questions):
    groups = defaultdict(list)
    for row in questions:
        groups[row["level"]].append(row)
    counts = {level: len(rows) * 100 // len(questions) for level, rows in groups.items()}
    remaining = 100 - sum(counts.values())
    levels = sorted(groups, key=lambda l: (-(len(groups[l]) * 100 % len(questions)), l))
    for level in levels[:remaining]:
        counts[level] += 1

    picked = []
    for level, rows in groups.items():
        def priority(row):
            return hashlib.sha256((SELECTION_KEY + row["unique_id"]).encode("utf-8")).hexdigest()
        picked.extend(sorted(rows, key=priority)[:counts[level]])
    return {row["unique_id"] for row in picked}, counts


def prepare(args):
    from main import get_prompts
    from src.math_core import last_boxed_only_string, remove_boxed
    from transformers import AutoTokenizer, GenerationConfig

    if digest(args.dataset) != DATASET_SHA or args.model.name != ASSET_NAME:
        raise ValueError("Expected the pinned MATH-500 and Llama assets")
    questions = [json.loads(line) for line in args.dataset.read_bytes().splitlines()]
    if len(questions) != 500 or len({row["unique_id"] for row in questions}) != 500:
        raise ValueError("Expected 500 distinct test questions")
    sample, counts = freeze_sample(questions)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    generation = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    if (generation.top_k, generation.repetition_penalty, tokenizer.eos_token_id) != (50, 1.0, 128009):
        raise ValueError("The pinned model sampling defaults differ from the historical settings")

    prompts = get_prompts()
    rows = []
    for index, row in enumerate(questions):
        problem = row["problem"]
        messages = [
            {"role": "system", "content": prompts["direct_unstructured_final_answer_system_prompt"]},
            {"role": "user", "content": prompts["direct_unstructured_final_answer_user_prompt"].format(problem=problem)},
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        answer = last_boxed_only_string(row["solution"])
        rows.append({
            "problem_id": index, "dataset_unique_id": row["unique_id"], "level": row["level"],
            "problem": problem, "gold": remove_boxed(answer) if answer else row["answer"], "prompt_token_ids": ids,
            "prompt_sha256": hashlib.sha256(encode(ids)).hexdigest(),
            # The historical Direct run used generation call indices 1, 3, ..., 19 for free-form answers.
            "seeds": [index + 500 * (2 * i + 1) for i in range(MODEL_CHOICES)],
        })
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_sha256": DATASET_SHA, "upstream_commit": UPSTREAM,
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "dtype": "bfloat16",
        "selection_key": SELECTION_KEY, "pad_token_id": tokenizer.pad_token_id,
        "decode": {"temperature": 0.7, "top_p": 0.9, "top_k": generation.top_k,
                   "repetition_penalty": generation.repetition_penalty,
                   "max_tokens": 1024, "eos_token_id": tokenizer.eos_token_id},
    }
    for phase, items in (
        ("pilot", [q for q in rows if q["dataset_unique_id"] in sample]),
        ("extension", [q for q in rows if q["dataset_unique_id"] not in sample]),
    ):
        path = args.output / f"{phase}_manifest.json"
        selected = Counter(row["level"] for row in items)
        if phase == "pilot" and (len(items) != 100 or dict(selected) != counts):
            raise ValueError("Wrong sample composition")
        # Store the token IDs without indenting each token on a separate line.
        path.write_bytes(encode({**config, "phase": phase, "levels": selected, "questions": items}))
        print(phase, len(items), dict(sorted(selected.items())), digest(path), flush=True)


def load_manifest(path):
    config = read(path)
    rows = config["questions"]
    if ((config["dataset_sha256"], config["upstream_commit"], config["model_revision"])
            != (DATASET_SHA, UPSTREAM, MODEL_REVISION)
            or [r["problem_id"] for r in rows] != sorted({r["problem_id"] for r in rows})
            or len({r["dataset_unique_id"] for r in rows}) != len(rows)):
        raise ValueError("Invalid question manifest")
    for row in rows:
        if (hashlib.sha256(encode(row["prompt_token_ids"])).hexdigest() != row["prompt_sha256"]
                or row["seeds"] != [row["problem_id"] + 500 * (2 * i + 1) for i in range(MODEL_CHOICES)]):
            raise ValueError(f"Unequal prompt or seed for question {row['problem_id']}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())

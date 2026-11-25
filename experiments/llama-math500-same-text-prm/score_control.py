"""Score identical paragraph text with one or many PRM separators."""

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
import time


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "llama-math500-step-prm-replay"
sys.path.insert(0, str(REFERENCE))

from prepare_pool import CASES, K, PRM_PROMPT, PRM_REVISION, SAMPLES, STEP_MARKER, digest, encode, read_pool


PARAGRAPH_BREAK = "\n\n"
INPUT_SHA256 = {
    "E2": "86162531efbe257cbedacaec7febddb8a6172f051afe4651d83bf8f4769c6ea1",
    "E4": "73c3f95e4ad26e7703c6068de73547dc4fe9b1bca23332254dd7e357c408ec3d",
}


def format_pair(steps):
    if not steps or any(STEP_MARKER in step for step in steps):
        raise ValueError("Expected nonempty steps without inline separators")
    whole = PARAGRAPH_BREAK.join(steps)
    step = (STEP_MARKER + PARAGRAPH_BREAK).join(steps)
    if step.replace(STEP_MARKER, "") != whole:
        raise ValueError("The two inputs differ beyond inserted PRM markers")
    # chat_tokens adds the same terminal marker to both strings.
    return whole, step


def tokenize_pairs(inputs, lane, scorer, step_id):
    from score_steps import chat_tokens

    tasks = []
    limit = scorer.model.config.max_position_embeddings
    for case in CASES:
        path = inputs / f"{case.lower()}_pool.jsonl.gz"
        if digest(path) != INPUT_SHA256[case]:
            raise ValueError(f"{case}: expected the PR #21 input pool")
        for row in read_pool(path)[lane::2]:
            if row["case"] != case:
                raise ValueError(f"{case}: wrong pool label")
            for candidate in row["candidates"]:
                steps = candidate["steps"]
                key = (case, row["problem_id"], candidate["iteration"])
                whole, step = format_pair(steps)
                a = chat_tokens(scorer.tokenizer, row["problem"], [whole])
                b = chat_tokens(scorer.tokenizer, row["problem"], [step])
                if (max(len(a), len(b)) > limit or a.count(step_id) != 1
                        or b.count(step_id) != len(steps)):
                    raise ValueError(f"{key}: PRM context length or marker mismatch")
                tasks.append((key, a, b, len(steps)))
    tasks.sort(key=lambda item: (max(len(item[1]), len(item[2])), item[0]))
    return tasks


def batches(tasks, size, budget):
    batch = []
    for task in tasks:
        width = max(len(task[1]), len(task[2]))
        if batch and (2 * (len(batch) + 1) > size or 2 * (len(batch) + 1) * width > budget):
            yield batch
            batch = []
        batch.append(task)
    if batch:
        yield batch


def score_pairs(scorer, step_id, batch):
    from score_steps import score_batch, torch

    flat = [(key, ids) for key, whole, step, _ in batch for ids in (whole, step)]
    try:
        values = score_batch(scorer, step_id, flat)
    except torch.OutOfMemoryError:
        if len(batch) == 1:
            raise
    else:
        result = []
        for (key, _, _, count), a, b in zip(batch, values[::2], values[1::2]):
            if len(a) != 1 or len(b) != count or any(not math.isfinite(x) or not 0 <= x <= 1 for x in a + b):
                raise ValueError(f"{key}: incomplete or invalid PRM rewards")
            result.append((*key, a[0], json.dumps(b, allow_nan=False)))
        return result
    torch.cuda.empty_cache()
    midpoint = len(batch) // 2
    print(f"Splitting a batch of {len(batch)} complete input pairs", flush=True)
    return score_pairs(scorer, step_id, batch[:midpoint]) + score_pairs(scorer, step_id, batch[midpoint:])


def run(args):
    from score_steps import PRMSelector, torch, transformers, write_json

    if args.batch_size < 2 or args.batch_tokens < 2 or not torch.cuda.is_available():
        raise ValueError("A CUDA device and enough room for at least one input pair are required")
    if args.model.name != f"Qwen--Qwen2.5-Math-PRM-7B--{PRM_REVISION}":
        raise ValueError("Expected the pinned Qwen Math PRM asset")
    args.output.mkdir(exist_ok=True, parents=True)
    started = time.monotonic()
    scorer = PRMSelector(str(args.model), device="cuda:0")
    step_id, = scorer.tokenizer.encode(STEP_MARKER, add_special_tokens=False)
    tasks = tokenize_pairs(args.inputs, args.lane, scorer, step_id)
    expected = len(CASES) * SAMPLES // 2 * K
    if len(tasks) != expected:
        raise ValueError("Incomplete paired candidate shard")
    config = {
        "lane": args.lane, "input_sha256": INPUT_SHA256, "prm_revision": PRM_REVISION,
        "system_prompt": PRM_PROMPT, "step_marker": STEP_MARKER, "paragraph_break": PARAGRAPH_BREAK,
        "batch_size": args.batch_size, "batch_tokens": args.batch_tokens,
        "dtype": str(scorer.model.dtype), "torch": torch.__version__, "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(), "attention": scorer.model.config._attn_implementation,
        "max_context": scorer.model.config.max_position_embeddings,
        "source_sha256": {
            "score_control.py": digest(Path(__file__)),
            "prepare_pool.py": digest(REFERENCE / "prepare_pool.py"),
            "score_steps.py": digest(REFERENCE / "score_steps.py"),
        },
    }
    print(f"GPU {args.lane}: validated {len(tasks)} input pairs; "
          f"longest sequence {max(len(ids) for _, a, b, _ in tasks for ids in (a, b))} tokens", flush=True)
    cfg = args.output / "config.json"
    with sqlite3.connect(args.output / "scores.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS scores (case_name TEXT, problem INTEGER, iteration INTEGER, "
            "whole REAL, steps TEXT, PRIMARY KEY(case_name, problem, iteration))"
        )
        done = set(db.execute("SELECT case_name, problem, iteration FROM scores"))
        if cfg.exists():
            if json.loads(cfg.read_bytes()) != config:
                raise ValueError("Checkpoint uses a different scorer configuration")
        else:
            if done:
                raise ValueError("Checkpoint lacks its scorer configuration")
            write_json(cfg, config)
        if not done.issubset({key for key, _, _, _ in tasks}):
            raise ValueError("Checkpoint includes an unexpected candidate")

        reported = len(done)
        for batch in batches(tasks, args.batch_size, args.batch_tokens):
            present = sum(key in done for key, _, _, _ in batch)
            if present == len(batch):
                continue
            if present:
                raise ValueError("Checkpoint contains a partial batch")
            scored = score_pairs(scorer, step_id, batch)
            with db:
                db.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?)", scored)
            done.update((case, index, iteration) for case, index, iteration, _, _ in scored)
            if len(done) - reported >= 200 or len(done) == expected:
                print(f"GPU {args.lane}: {len(done)}/{expected} pairs in {time.monotonic()-started:.1f}s", flush=True)
                reported = len(done)

        target = args.output / "paired_scores.jsonl"
        temporary = target.with_suffix(".tmp")
        sql = "SELECT * FROM scores ORDER BY case_name, problem, iteration"
        with temporary.open("wb") as file:
            for case, index, iteration, whole, rewards in db.execute(sql):
                file.write(encode({
                    "case": case, "problem_id": index, "iteration": iteration,
                    "whole_score": whole, "step_rewards": json.loads(rewards),
                }))
        temporary.replace(target)

    write_json(args.output / "runtime.json", {
        "candidate_pairs": len(done), "seconds": time.monotonic() - started,
        "whole_input_tokens": sum(len(a) for _, a, _, _ in tasks),
        "step_input_tokens": sum(len(b) for _, _, b, _ in tasks),
        "scores_sha256": digest(target), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=REFERENCE / "inputs")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    run(parser.parse_args())

"""Score the fixed trajectories at every step boundary with the original PRM."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import torch
import transformers
from src.help_functions.prm_selector import PRMSelector

from prepare_pool import K, PRM_PROMPT, PRM_REVISION, STEP_MARKER, digest, encode, read_pool, winner


def chat_tokens(tokenizer, problem, steps):
    messages = [
        {"role": "system", "content": PRM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": STEP_MARKER.join(steps) + STEP_MARKER},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.encode(prompt)


def tokenize_pool(rows, scorer, step_id):
    tasks = []
    total = 0
    limit = scorer.model.config.max_position_embeddings
    for row in rows:
        for candidate in row["candidates"]:
            steps = candidate["steps"]
            key = (row["problem_id"], candidate["iteration"])
            tokens = chat_tokens(scorer.tokenizer, row["problem"], steps)
            if tokens.count(step_id) != len(steps) or len(tokens) > limit:
                raise ValueError(f"{row['case']}/{key}: step markers or context length {len(tokens)}/{limit}")
            tasks.append((key, tokens))
            total += len(steps)
    tasks.sort(key=lambda item: (len(item[1]), item[0]))
    print(f"Tokenized {len(tasks)} candidates, {total} steps, longest input {len(tasks[-1][1])} tokens", flush=True)
    return tasks


def batch_tasks(tasks, size, max_tokens):
    batch = []
    for task in tasks:
        if batch and (len(batch) >= size or (len(batch) + 1) * len(task[1]) > max_tokens):
            yield batch
            batch = []
        batch.append(task)
    if batch:
        yield batch


@torch.inference_mode()
def score_batch(scorer, step_id, batch):
    width = max(len(tokens) for _, tokens in batch)
    pad = scorer.tokenizer.pad_token_id
    device = scorer.model.device
    ids = torch.tensor([tokens + [pad] * (width - len(tokens)) for _, tokens in batch], device=device)
    valid = torch.tensor([[1] * len(tokens) + [0] * (width - len(tokens)) for _, tokens in batch], device=device)
    logits = scorer.model(input_ids=ids, attention_mask=valid, use_cache=False)[0]
    boundaries = (ids == step_id) & valid.bool()
    # Keep the original score precision and pick the positive class at each separator.
    positive = logits.softmax(dim=-1)[..., 1]
    return [row[mask].tolist() for row, mask in zip(positive, boundaries)]


def score_with_split(scorer, step_id, batch):
    try:
        return score_batch(scorer, step_id, batch)
    except torch.OutOfMemoryError:
        if len(batch) == 1:
            raise
    torch.cuda.empty_cache()
    left = len(batch) // 2
    print(f"Splitting an oversized batch of {len(batch)} candidates", flush=True)
    return score_with_split(scorer, step_id, batch[:left]) + score_with_split(scorer, step_id, batch[left:])


def legacy_check(scorer, step_id, history, case, size, budget):
    tasks, old = [], {}
    for index in (0, 49, 99, 199, 299, 399, 499):
        path = history / f"formal/math/gpu-{index % 2}/{case}/problems/{index:03d}/result.json"
        native = json.loads(path.read_bytes())
        for item in native["iterations"]:
            key = (index, item["iteration"])
            steps = scorer.combine_fields(item)
            tokens = chat_tokens(scorer.tokenizer, native["problem"], steps)
            if tokens.count(step_id) != 1:
                raise ValueError(f"{case}/{key}: legacy canary did not have exactly one separator")
            tasks.append((key, tokens))
            old[key] = item["prm_scores"]["mean_reward"]
    tasks.sort(key=lambda item: (len(item[1]), item[0]))
    scored = {}
    for batch in batch_tasks(tasks, size, budget):
        for (key, _), values in zip(batch, score_with_split(scorer, step_id, batch)):
            score, = values
            scored[key] = score
    differences = [abs(scored[key] - score) for key, score in old.items()]
    selected = 0
    indices = sorted({index for index, _ in old})
    for index in indices:
        keys = [(index, i) for i in range(1, K + 1)]
        selected += winner([scored[k] for k in keys]) == winner([old[k] for k in keys])
    result = {
        "questions": len(indices), "same_selected_iterations": selected,
        "candidates": len(differences), "exact_score_matches": differences.count(0.0),
        "max_absolute_score_delta": max(differences),
    }
    print("Historical single-block canary", result, flush=True)
    return result


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def run(args):
    rows = read_pool(args.pool)
    case = rows[0]["case"]
    if args.model.name != f"Qwen--Qwen2.5-Math-PRM-7B--{PRM_REVISION}":
        raise ValueError("Use the pinned Qwen Math PRM asset directory")
    if not torch.cuda.is_available() or min(args.batch_size, args.batch_tokens) < 1:
        raise ValueError("A CUDA device and positive batch limits are required")

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    scorer = PRMSelector(str(args.model), device="cuda:0")
    step_id, = scorer.tokenizer.encode(STEP_MARKER, add_special_tokens=False)
    tasks = tokenize_pool(rows, scorer, step_id)
    config = {
        "case": case, "input_sha256": digest(args.pool), "prm_revision": PRM_REVISION,
        "system_prompt": PRM_PROMPT, "marker": STEP_MARKER, "batch_size": args.batch_size,
        "batch_tokens": args.batch_tokens, "torch": torch.__version__, "transformers": transformers.__version__,
        "dtype": str(scorer.model.dtype), "device": torch.cuda.get_device_name(),
        "attention": scorer.model.config._attn_implementation, "max_prm_context": scorer.model.config.max_position_embeddings,
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in ("prepare_pool.py", "score_steps.py")},
    }
    cfg = args.output / "config.json"
    with sqlite3.connect(args.output / "scores.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS scores (problem INTEGER, iteration INTEGER, rewards TEXT, "
                   "PRIMARY KEY(problem, iteration))")
        done = {(a, b) for a, b in db.execute("SELECT problem, iteration FROM scores")}
        if cfg.exists() and json.loads(cfg.read_bytes()) != config:
            raise ValueError("Existing scores use a different input, scorer or batch configuration")
        if not cfg.exists():
            if done:
                raise ValueError("Existing candidate scores are missing their configuration")
            write_json(cfg, config)
        canary = legacy_check(scorer, step_id, args.history, case, args.batch_size, args.batch_tokens) if args.history else None

        last_report = len(done)
        for batch in batch_tasks(tasks, args.batch_size, args.batch_tokens):
            present = sum(key in done for key, _ in batch)
            if present == len(batch):
                continue
            if present:
                raise ValueError("Checkpoint contains part of a batch")
            rewards = score_with_split(scorer, step_id, batch)
            with db:
                db.executemany("INSERT INTO scores VALUES (?, ?, ?)", [
                    (index, iteration, json.dumps(step_scores, allow_nan=False))
                    for ((index, iteration), _), step_scores in zip(batch, rewards)
                ])
            done.update(key for key, _ in batch)
            if len(done) - last_report >= 250 or len(done) == len(tasks):
                print(f"{case} {len(done)}/{len(tasks)} candidates scored in {time.monotonic() - started:.1f}s", flush=True)
                last_report = len(done)

        out = args.output / "step_scores.jsonl"
        temporary = out.with_suffix(".tmp")
        with temporary.open("wb") as file:
            for index, iteration, values in db.execute("SELECT * FROM scores ORDER BY problem, iteration"):
                file.write(encode({"problem_id": index, "iteration": iteration, "step_rewards": json.loads(values)}))
        temporary.replace(out)
    write_json(args.output / "runtime.json", {
        "candidates": len(done), "seconds": time.monotonic() - started,
        "input_tokens": sum(len(tokens) for _, tokens in tasks), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "score_sha256": digest(out), "legacy_canary": canary,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path, help="Optional replay of a fixed subset of historical one-block scores")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    run(parser.parse_args())

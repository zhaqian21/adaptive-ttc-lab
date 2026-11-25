"""Generate a fixed number of free-form E2 candidates with either backend."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from unittest.mock import patch

import requests
from transformers import AutoTokenizer

from prepare_sample import ASSET_NAME, MODEL_CHOICES, MODEL_ID, digest, encode, load_manifest, read, write_json


THREAD = threading.local()


def record(row, iteration, ids, finish, tokenizer):
    return {
        "problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"],
        "iteration": iteration, "seed": row["seeds"][iteration - 1], "prompt_sha256": row["prompt_sha256"],
        "token_ids_sha256": hashlib.sha256(encode(ids)).hexdigest(), "completion_tokens": len(ids),
        "finish_reason": finish, "completion": tokenizer.decode(ids, skip_special_tokens=True).strip(),
    }


def vllm_request(task, tokenizer, config, endpoint):
    row, i = task
    decode = config["decode"]
    seed = row["seeds"][i - 1]
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
        THREAD.session.trust_env = False
    payload = {
        "model": MODEL_ID, "prompt": row["prompt_token_ids"], "seed": seed,
        **{key: value for key, value in decode.items() if key != "eos_token_id"},
        "stop_token_ids": [decode["eos_token_id"]], "ignore_eos": True,
        "skip_special_tokens": True, "return_token_ids": True,
    }
    for attempt in range(3):
        try:
            response = THREAD.session.post(endpoint + "/v1/completions", json=payload, timeout=(3, 420))
            response.raise_for_status()
            body = response.json()
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(1 << attempt)
    choice, = body["choices"]
    ids = choice["token_ids"]
    reason = choice["finish_reason"]
    if (choice["prompt_token_ids"] != row["prompt_token_ids"]
            or len(ids) != body["usage"]["completion_tokens"]
            or not 0 < len(ids) <= decode["max_tokens"]):
        raise ValueError(f"{row['problem_id']}/{i}: vLLM prompt or token count mismatch")
    if reason == "length":
        if len(ids) != decode["max_tokens"]:
            raise ValueError("vLLM length stop before the output cap")
    elif reason != "stop" or (choice.get("stop_reason") != decode["eos_token_id"] and ids[-1] != decode["eos_token_id"]):
        raise ValueError(f"Unexpected vLLM stop: {reason!r} / {choice.get('stop_reason')!r}")
    return record(row, i, ids, "length" if reason == "length" else "eos", tokenizer)


class TransformersGenerator:
    def __init__(self, model_dir, tokenizer, config):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.tokenizer = tokenizer
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, device_map="cuda:0", torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        ).eval()
        self.runtime = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "attention": self.model.config._attn_implementation,
            "device": torch.cuda.get_device_name(), "dtype": str(self.model.dtype),
            "sampling": "native Transformers generate, independent torch multinomial RNG per row",
        }

    def generate(self, tasks):
        torch = self.torch
        decode = self.config["decode"]
        raw = [row["prompt_token_ids"] for row, _ in tasks]
        width = max(map(len, raw))
        pad = self.config["pad_token_id"]
        eos = decode["eos_token_id"]
        device = self.model.device
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in raw], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in raw], device=device)
        generators = [torch.Generator(device=device).manual_seed(row["seeds"][i - 1]) for row, i in tasks]
        multinomial = torch.multinomial
        draws = 0

        def seeded_rows(probs, num_samples, *args, **kwargs):
            nonlocal draws
            if probs.ndim != 2 or probs.shape[0] != len(generators) or num_samples != 1 or args or kwargs:
                raise RuntimeError("Unexpected Transformers multinomial call")
            draws += 1
            return torch.stack([multinomial(p, 1, generator=rng) for p, rng in zip(probs, generators)])

        with torch.inference_mode(), patch.object(torch, "multinomial", seeded_rows):
            output = self.model.generate(
                input_ids=ids, attention_mask=mask, do_sample=True, num_beams=1,
                max_new_tokens=decode["max_tokens"], eos_token_id=eos, pad_token_id=pad,
                temperature=decode["temperature"], top_p=decode["top_p"], top_k=decode["top_k"],
                repetition_penalty=decode["repetition_penalty"], use_cache=True,
            )
        if draws != output.shape[1] - width:
            raise ValueError("Per-row seed was bypassed during Transformers generation")
        records = []
        for (row, i), tokens in zip(tasks, output[:, width:].cpu().tolist()):
            if eos in tokens:
                tokens = tokens[:tokens.index(eos) + 1]
                finish = "eos"
            else:
                if len(tokens) != decode["max_tokens"]:
                    raise ValueError("Transformers stopped before EOS and the output cap")
                finish = "length"
            records.append(record(row, i, tokens, finish, self.tokenizer))
        return records

    def generate_with_retry(self, tasks):
        try:
            return self.generate(tasks)
        except self.torch.OutOfMemoryError:
            if len(tasks) == 1:
                raise
        self.torch.cuda.empty_cache()
        half = len(tasks) // 2
        print(f"Splitting a Transformers batch of {len(tasks)} candidates", flush=True)
        return self.generate_with_retry(tasks[:half]) + self.generate_with_retry(tasks[half:])


def run(args):
    if args.model.name != ASSET_NAME:
        raise ValueError("Expected the pinned Llama model directory")
    config = load_manifest(args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if (tokenizer.pad_token_id, tokenizer.eos_token_id) != (config["pad_token_id"], config["decode"]["eos_token_id"]):
        raise ValueError("Model tokenizer differs from the frozen manifest")
    tasks = [(row, i) for row in config["questions"] for i in range(1, MODEL_CHOICES + 1)]
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    cfg = {
        "backend": args.backend, "manifest_sha256": digest(args.manifest), "decode": config["decode"],
        "model_revision": config["model_revision"], "batch_or_concurrency": args.batch,
        "source_sha256": digest(Path(__file__)),
    }
    if args.backend == "transformers":
        native = TransformersGenerator(args.model, tokenizer, config)
        cfg.update(native.runtime)
    else:
        cfg.update({"vllm": args.vllm_version, "sampling": "independent explicit seed on each raw-token request"})

    with sqlite3.connect(args.output / "generation.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS completion (problem INTEGER, iteration INTEGER, row BLOB, "
                   "PRIMARY KEY (problem, iteration))")
        saved = args.output / "config.json"
        done = set(db.execute("SELECT problem, iteration FROM completion"))
        if saved.exists() and read(saved) != cfg:
            raise ValueError("Cannot resume generation under a different setup")
        if not saved.exists():
            if done:
                raise ValueError("Completed rows without a configuration")
            write_json(saved, cfg)
        ordered = {(r["problem_id"], i) for r, i in tasks}
        if not done.issubset(ordered):
            raise ValueError("Unexpected completed request in checkpoint")
        todo = [(r, i) for r, i in tasks if (r["problem_id"], i) not in done]
        total = len(tasks)
        print(f"{args.backend}: {len(done)}/{total} candidates in checkpoint", flush=True)

        def save(rows):
            with db:
                db.executemany("INSERT INTO completion VALUES (?, ?, ?)", [
                    (r["problem_id"], r["iteration"], encode(r)) for r in rows
                ])
            done.update((r["problem_id"], r["iteration"]) for r in rows)

        if args.backend == "transformers":
            for pos in range(0, len(todo), args.batch):
                batch = todo[pos:pos + args.batch]
                save(native.generate_with_retry(batch))
                print(f"{args.backend}: {len(done)}/{total} candidates in {time.monotonic()-start:.1f}s", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.batch) as pool:
                for pos in range(0, len(todo), args.batch):
                    batch = todo[pos:pos + args.batch]
                    save(list(pool.map(lambda task: vllm_request(task, tokenizer, config, args.endpoint), batch)))
                    print(f"{args.backend}: {len(done)}/{total} candidates in {time.monotonic()-start:.1f}s", flush=True)
        if done != ordered:
            raise ValueError("Incomplete generation archive")
        output = args.output / "candidates.jsonl"
        pending = output.with_suffix(".tmp")
        with pending.open("wb") as file:
            for payload, in db.execute("SELECT row FROM completion ORDER BY problem, iteration"):
                file.write(payload)
        pending.replace(output)
    write_json(args.output / "runtime.json", {
        "candidates": len(done), "seconds": time.monotonic() - start,
        "candidates_sha256": digest(output),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("transformers", "vllm"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:22670")
    parser.add_argument("--vllm-version", default="0.33.0")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    run(args)

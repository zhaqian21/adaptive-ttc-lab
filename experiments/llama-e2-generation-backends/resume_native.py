"""Split the remaining native generation work after the vLLM lane finishes."""

import argparse
from pathlib import Path
import sqlite3
import time

from generate_candidates import TransformersGenerator, AutoTokenizer
from prepare_sample import ASSET_NAME, MODEL_CHOICES, digest, encode, load_manifest, read, write_json


def keys_for(rows):
    return [(r["problem_id"], i) for r in rows for i in range(1, MODEL_CHOICES + 1)]


def checkpoint(work):
    db_file = work / "transformers" / "generation.sqlite3"
    with sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True) as db:
        return {(a, b): value for a, b, value in
                db.execute("SELECT problem,iteration,row FROM completion ORDER BY problem,iteration")}


def plan(args, manifest):
    retained = checkpoint(args.work)
    expected = keys_for(manifest["questions"])
    start = read(args.work / "transformers" / "config.json")
    if not set(retained).issubset(expected) or start["manifest_sha256"] != digest(args.manifest):
        raise ValueError("The native checkpoint is from a different manifest")
    todo = [key for key in expected if key not in retained]
    # The second GPU has more memory after the temporary vLLM service exits.
    cutoff = ((len(todo) * 2 // 5 + 16) // 32) * 32
    write_json(args.work / "native_plan.json", {
        "manifest_sha256": digest(args.manifest), "initial_transformers_config": start,
        "retained_keys": list(retained), "retained_sha256": sha(b"".join(retained.values())),
        "lanes": {"0": {"batch": 32, "keys": todo[:cutoff]}, "1": {"batch": 64, "keys": todo[cutoff:]}},
    })
    print("Retained", len(retained), "candidates; remaining tasks per GPU", cutoff, len(todo)-cutoff, flush=True)


def sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def read_plan(args):
    task_plan = read(args.work / "native_plan.json")
    if task_plan["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Task plan and manifest differ")
    return task_plan


def worker(args, manifest):
    if args.model is None or args.model.name != ASSET_NAME or args.lane is None:
        raise ValueError("Expected the pinned model and lane")
    task_plan = read_plan(args)
    lane = task_plan["lanes"][str(args.lane)]
    by_id = {r["problem_id"]: r for r in manifest["questions"]}
    tasks = [(by_id[index], i) for index, i in lane["keys"]]
    out = args.work / f"native-gpu-{args.lane}"
    out.mkdir(exist_ok=True)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    native = TransformersGenerator(args.model, tokenizer, manifest)
    initial = task_plan["initial_transformers_config"]
    if any(initial[key] != value for key, value in native.runtime.items()):
        raise ValueError("Native worker runtime differs from the original Transformers lane")

    with sqlite3.connect(out / "generation.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS completion (problem INTEGER, iteration INTEGER, row BLOB, "
                   "PRIMARY KEY(problem, iteration))")
        done = set(db.execute("SELECT problem,iteration FROM completion"))
        expected = {tuple(key) for key in lane["keys"]}
        if not done.issubset(expected):
            raise ValueError("Native shard includes an unassigned candidate")
        todo = [(r, i) for r, i in tasks if (r["problem_id"], i) not in done]
        for pos in range(0, len(todo), lane["batch"]):
            records = native.generate_with_retry(todo[pos:pos + lane["batch"]])
            with db:
                db.executemany("INSERT INTO completion VALUES (?,?,?)",
                               [(r["problem_id"], r["iteration"], encode(r)) for r in records])
            done.update((r["problem_id"], r["iteration"]) for r in records)
            print(f"native GPU {args.lane}: {len(done)}/{len(tasks)} candidates in {time.monotonic()-start:.1f}s", flush=True)
        if done != expected:
            raise ValueError("Incomplete native shard")
        path = out / "candidates.jsonl"
        with path.with_suffix(".tmp").open("wb") as stream:
            for row, in db.execute("SELECT row FROM completion ORDER BY problem,iteration"):
                stream.write(row)
        path.with_suffix(".tmp").replace(path)
    write_json(out / "runtime.json", {
        "candidates": len(done), "seconds": time.monotonic()-start, "candidate_sha256": digest(path),
        "peak_allocated_bytes": native.torch.cuda.max_memory_allocated(), "batch": lane["batch"],
    })


def merge(args, manifest):
    import json

    task_plan = read_plan(args)
    retained = checkpoint(args.work)
    if ([list(key) for key in retained] != task_plan["retained_keys"]
            or sha(b"".join(retained.values())) != task_plan["retained_sha256"]):
        raise ValueError("Retained Transformers candidates changed")
    runtimes = []
    for lane in (0, 1):
        folder = args.work / f"native-gpu-{lane}"
        path = folder / "candidates.jsonl"
        runtime = read(folder / "runtime.json")
        payloads = path.read_bytes().splitlines(keepends=True)
        rows = [json.loads(row) for row in payloads]
        keys = [(row["problem_id"], row["iteration"]) for row in rows]
        if (keys != [tuple(key) for key in task_plan["lanes"][str(lane)]["keys"]]
                or set(retained).intersection(keys) or digest(path) != runtime["candidate_sha256"]):
            raise ValueError(f"Native shard {lane} does not match the planned tasks")
        retained.update(zip(keys, payloads))
        runtimes.append(runtime)
    expected = keys_for(manifest["questions"])
    if sorted(retained) != expected:
        raise ValueError("Incomplete merged native candidate pool")

    folder = args.work / "transformers"
    path = folder / "candidates.jsonl"
    path.with_suffix(".tmp").write_bytes(b"".join(retained[key] for key in expected))
    path.with_suffix(".tmp").replace(path)
    config = task_plan["initial_transformers_config"]
    config["native_resume_plan_sha256"] = digest(args.work / "native_plan.json")
    config["native_resume_source_sha256"] = digest(Path(__file__))
    write_json(folder / "config.json", config)
    write_json(folder / "runtime.json", {
        "candidates": len(retained), "candidates_sha256": digest(path),
        "retained_candidates": len(task_plan["retained_keys"]), "resume_workers": runtimes,
    })
    print("Merged and audited", len(retained), "native candidates", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "merge"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    args.work = args.work.resolve()
    {"plan": plan, "run": worker, "merge": merge}[args.action](args, load_manifest(args.manifest))

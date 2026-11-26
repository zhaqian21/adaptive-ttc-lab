"""Continue an E4 run across two vLLM workers with per-question leases."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import yaml

if Path(__file__).with_name("run_vllm_cases_parallel_base.py").is_file():
    import run_vllm_cases_parallel_base as core
else:
    import run_vllm_cases as core


COUNT = 500
MAX_ATTEMPTS = 3


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path, text):
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def result_path(output, index):
    return output / "problems" / f"{index:03d}" / "result.json"


def prepare(batch, output, shared, config):
    precondition = read(batch / "E3" / "audit.json")
    assert (batch / "E3.status").read_text().strip() == "verified"
    assert precondition["samples"] == COUNT and precondition["summary_verified"] and precondition["pinned_dataset_checked"]
    path = shared / "work.json"
    with (shared / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return
        preserved = {}
        for index in range(COUNT):
            saved = result_path(output, index)
            if saved.is_file():
                preserved[str(index)] = sha256(saved)
        reference = read(batch / "E1-vllm" / "requests.json")
        manifest = {
            "case": "E4", "activated_at_unix": time.time(), "samples": COUNT,
            "gpu_uuids": {"0": reference["gpu_uuid"], "1": config["reproduction"]["runtime"]["cuda_visible_devices"]},
            "dataset_sha256": reference["dataset_sha256"], "initial_config_sha256": sha256(output / "config.yaml"),
            "base_runner_sha256": sha256(Path(core.__file__)), "parallel_runner_sha256": sha256(Path(__file__)),
            "preserved_result_sha256": preserved,
            "pending_ids": [index for index in range(COUNT) if str(index) not in preserved],
        }
        atomic_json(path, manifest)
        print(f"Preserved {len(preserved)} GPU 1 checkpoints. {len(manifest['pending_ids'])} questions enter the shared pool.", flush=True)


def solve_leased(question, config, output, leases, client_options):
    index = question["problem_id"]
    with (leases / f"{index:03d}.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy"
        if result_path(output, index).is_file():
            return "committed"
        journal = leases / f"{index:03d}.json"
        attempts = read(journal) if journal.is_file() else []
        if sum(attempt.get("status") == "failed" for attempt in attempts) >= MAX_ATTEMPTS:
            return "exhausted"
        for attempt in attempts:
            attempt.setdefault("status", "interrupted")
        entry = {"gpu": config["reproduction"]["physical_gpu"], "started_unix": time.time()}
        attempts.append(entry)
        atomic_json(journal, attempts)
        try:
            core.solve_question(question, config, output, *client_options)
        except Exception as error:
            entry.update(status="failed", message=str(error))
            atomic_json(journal, attempts)
            print(f"GPU {entry['gpu']} E4 problem {index} failed an attempt: {error}", flush=True)
            return "failed"
        entry.update(status="committed", finished_unix=time.time())
        atomic_json(journal, attempts)
        print(f"GPU {entry['gpu']} committed E4 problem {index}", flush=True)
        return "committed"


def complete(output, shared, manifest, config, canonical, questions):
    with (shared / ".merge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / "audit.json").is_file():
            return
        predictions = []
        counts = {"0": 0, "1": 0}
        preserved = manifest["preserved_result_sha256"]
        for index in range(COUNT):
            path = result_path(output, index)
            row = read(path)
            if str(index) in preserved:
                assert sha256(path) == preserved[str(index)], f"Preserved checkpoint changed: {index}"
                gpu = 1
            else:
                gpu = row["observation"]["physical_gpu"]
                assert gpu in (0, 1)
                assert row["observation"]["gpu_uuid"] == manifest["gpu_uuids"][str(gpu)]
            counts[str(gpu)] += 1
            predictions.append({"problem_id": row["problem_id"], "dataset_unique_id": row["dataset_unique_id"],
                                "gpu": gpu, "predicted_answer": row["final_predicted_answer"],
                                "is_correct": row["final_is_correct"], "selected_iteration": row["selected_iteration_idx"]})
        elapsed = time.time() - manifest["activated_at_unix"]
        summary = core.summarize(output, config, questions, elapsed)
        originals = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()]
        evidence = core.audit(output, originals)
        assert evidence["samples"] == COUNT and evidence["native_files_verified"] and evidence["pinned_dataset_checked"]
        atomic_text(output / "merged_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        merged = {**evidence, "backend": "vllm", "gpu_questions": counts,
                  "preserved_gpu1_checkpoints": len(preserved), "dual_gpu_elapsed_seconds": elapsed}
        batch = output.parent
        atomic_json(batch / "E4.audit.json", merged)
        atomic_text(batch / "E4.status", "verified\n")
        subprocess.run([sys.executable, str(batch / "compare-vllm.py"), str(batch)], check=True)
        atomic_json(output / "audit.json", merged)
        print(f"Merged and audited {COUNT} questions, GPU 0: {counts['0']}, GPU 1: {counts['1']}, correct: {summary['correct']}", flush=True)


def run(args):
    batch = Path(__file__).resolve().parent.parent
    output, shared = batch / "E4", batch / "E4-parallel"
    assert args.endpoint == f"http://127.0.0.1:{18409 if args.gpu == 0 else 18411}"
    config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    reproduction = config["reproduction"]
    assert reproduction["case"] == "E4" and config["eval"]["max_problems"] == COUNT
    assert config["eval"]["num_iterations"] == 10 and reproduction["num_workers"] == 32
    assert reproduction["runner_sha256"] == sha256(Path(core.__file__))
    assert args.vllm_version == reproduction["vllm_version"] and args.served_model == reproduction["served_model"]
    canonical = core.runner.asset_path(core.runner.ASSETS[0]) / "test.jsonl"
    assert sha256(canonical) == reproduction["dataset_sha256"]
    if args.mode == "coordinator":
        assert args.gpu == 1 and (shared / "gpu0.ready").is_file()
        prepare(batch, output, shared, config)
    manifest = read(shared / "work.json")
    runtime = core.runner.require_h20()
    assert runtime["cuda_visible_devices"] == manifest["gpu_uuids"][str(args.gpu)]
    assert sha256(output / "config.yaml") == manifest["initial_config_sha256"]
    runner_hash = sha256(Path(__file__))
    if runner_hash == manifest["parallel_runner_sha256"]:
        assert args.workers == 32
    else:
        profile = read(shared / "tuning" / "worker-profiles.json")
        assert profile["base_parallel_runner_sha256"] == manifest["parallel_runner_sha256"]
        assert profile["tuned_parallel_runner_sha256"] == runner_hash and args.workers in profile["worker_counts"]
    leases = shared / "leases"
    leases.mkdir(exist_ok=True)
    questions = core.runner.dataset(COUNT)
    assert [row["problem_id"] for row in questions] == list(range(COUNT))
    reproduction.update(physical_gpu=args.gpu)
    order = sorted(manifest["pending_ids"], reverse=args.gpu == 0)
    stamp = time.time_ns()
    metadata = {"gpu": args.gpu, "runtime": runtime, "endpoint": args.endpoint, "workers": args.workers,
                "started_unix": stamp / 1_000_000_000, "parallel_runner_sha256": runner_hash}
    (shared / "tuning").mkdir(exist_ok=True)
    atomic_json(shared / "tuning" / f"worker-{args.gpu}-{stamp}.json", metadata)
    atomic_json(shared / f"worker-{args.gpu}.json", metadata)
    client_options = args.endpoint, args.served_model, reproduction["decode"]
    prm = core.ConcurrentPRM()
    upstream = core.runner.upstream
    original_write = core.runner.write_json

    def write_with_device(path, row):
        if path.name == "result.json" and "observation" in row:
            row["observation"].update(physical_gpu=args.gpu, gpu_uuid=runtime["cuda_visible_devices"],
                                      worker_slots=args.workers, parallel_runner_sha256=runner_hash)
        original_write(path, row)

    with ExitStack() as stack:
        stack.enter_context(patch.object(upstream, "UniversalAgent", core.PaperAgent))
        stack.enter_context(patch.object(upstream, "create_llm_client", lambda _settings: core.context.client))
        stack.enter_context(patch.object(upstream, "PRMSelector", lambda: prm))
        stack.enter_context(patch.object(core.runner.universal_agent, "get_compute_strategy", core.strategy_factory(prm)))
        stack.enter_context(patch.object(core.runner, "write_json", write_with_device))
        workers = stack.enter_context(ThreadPoolExecutor(max_workers=args.workers))
        while True:
            missing = [index for index in order if not result_path(output, index).is_file()]
            if not missing:
                break
            jobs = [workers.submit(solve_leased, questions[index], config, output, leases, client_options) for index in missing]
            states = [future.result() for future in as_completed(jobs)]
            if all(state == "exhausted" for state in states):
                raise RuntimeError("Some E4 questions exhausted all retries; inspect the lease journals")
            if all(state in {"busy", "exhausted"} for state in states):
                time.sleep(3)
    complete(output, shared, manifest, config, canonical, questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("coordinator", "worker"))
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--workers", type=int, choices=(32, 64), default=32)
    run(parser.parse_args())

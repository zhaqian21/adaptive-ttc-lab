"""Run the pinned Llama inference policies on one allocated H20."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import compare
import llama_table1 as services
import matched
import run_adaptive_k as adaptive
import run_amc23 as amc
import run_math500 as revision
import run_modules as routing
from audit_math500 import load, shard_files, summarize_policies, validate_result


EXPERIMENTS = Path(__file__).resolve().parents[1]
STAGES = ("revision-math", "revision-amc23", "paired", "single-prm", "answer-prm-sum",
          "factorial", "g", "f", "fg")


def load_auditor(directory):
    path = EXPERIMENTS / directory / "audit.py"
    spec = importlib.util.spec_from_file_location(directory, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit


def amc_benchmark(args):
    questions = amc.load_amc23(args.dataset_file)
    if args.sample:
        per_shard = args.sample // 2
        questions = sorted(amc.for_shard(questions, 0, per_shard) + amc.for_shard(questions, 1, per_shard),
                           key=lambda question: question["problem_id"])
    return questions, {
        "dataset": amc.DATASET, "dataset_sha256": amc.DATASET["sha256"],
        "benchmark_source_sha256": hashlib.sha256(Path(amc.__file__).read_bytes()).hexdigest(),
        "sample": args.sample,
    }


def audit_revision(root, args, seconds):
    config, execution, files = shard_files(root / "shard-0", 0, 1)
    questions = (amc_benchmark(args)[0] if args.dataset_file is not None
                 else revision.baseline.dataset_for_run(args))
    expected = {q["problem_id"]: q for q in questions}
    if execution["failures"] or set(expected) != set(files):
        raise ValueError("Revision checkpoint coverage is incomplete")
    evaluator = revision.runner.upstream.get_evaluator("MATH")
    rows = []
    for index, path in sorted(files.items()):
        row = load(path)
        gold = expected[index]["gold_answer_extracted"]
        validate_result(row, config, index, expected[index]["dataset_unique_id"])
        for arm in row["policies"].values():
            candidates = arm["candidates"]
            labels = [bool(evaluator._check_answers_equiv(c["prediction"], gold)) for c in candidates]
            if labels != [c["correct"] for c in candidates]:
                raise ValueError(f"Wrong revision candidate label on row {index}")
            arm.update(has_correct_candidate=any(labels), prm_missed_correct_candidate=any(labels) and not arm["correct"])
        rows.append(row)
    policies = summarize_policies(rows)
    for name, arm in policies.items():
        arm["candidate_pool_correct"] = sum(row["policies"][name]["has_correct_candidate"] for row in rows)
        arm["prm_missed_correct"] = sum(row["policies"][name]["prm_missed_correct_candidate"] for row in rows)
        if arm["candidate_pool_correct"] != arm["correct"] + arm["prm_missed_correct"]:
            raise ValueError(f"Inconsistent candidate coverage for {name}")
    output = root / "export"
    output.mkdir(exist_ok=True)
    raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode()
    (output / "predictions.jsonl").write_bytes(raw)
    revision.baseline.write(output / "summary.json", {
        "samples": len(rows), "formal": args.sample is None, "wall_seconds": seconds,
        "policies": policies, "configs": [config], "executions": [execution],
        "predictions_sha256": hashlib.sha256(raw).hexdigest(),
    })


def run_stage(root, name, sample, workers, ports, dataset_file):
    root.mkdir(parents=True, exist_ok=True)
    stage_dir = root / name
    shard = stage_dir / "shard-0"
    stage_dir.mkdir(exist_ok=True)
    args = SimpleNamespace(
        output=shard, case="E4", priority_file=None, port=ports[0], prm_port=ports[1],
        context=32768, model=services.MODEL[1], vllm_version="0.33.0", workers=workers,
        seed=0, token_budget=125000, initial_roots=3, only_row=None, shards=1, shard=0,
        sample=sample, dataset_file=dataset_file if name == "revision-amc23" else None,
        mode="paired" if name == "paired" else "adaptive",
        selector="answer_prm_sum" if name == "answer-prm-sum" else "single_prm",
    )
    if name.startswith("revision-"):
        def run():
            return revision.run(args, load_benchmark=amc_benchmark if args.dataset_file is not None else None)

        def audit(path, elapsed):
            return audit_revision(path, args, elapsed)
    elif name in ("paired", "single-prm", "answer-prm-sum"):
        def run():
            return adaptive.run(args)

        shard_audit = load_auditor("math500-adaptive-k")

        def audit(path, elapsed):
            return shard_audit(path, elapsed, shards=1)
    else:
        args.method = name

        def run():
            return routing.run(args)

        shard_audit = load_auditor("math500-prm-routing")

        def audit(path, elapsed):
            return shard_audit(path, elapsed, shards=1)

    attempts_path = stage_dir / "attempts.json"
    attempts = load(attempts_path) if attempts_path.exists() else []
    for _ in range(2):
        attempt = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "seconds": None, "exit": None}
        attempts.append(attempt)
        revision.baseline.write(attempts_path, attempts)
        start = time.perf_counter()
        code = run()
        attempt.update(seconds=time.perf_counter() - start, exit=code)
        revision.baseline.write(attempts_path, attempts)
        if code == 0:
            break
        execution = shard / "execution.json"
        first = shard / "first-attempt.json"
        if not first.exists() and execution.exists():
            first.write_bytes(execution.read_bytes())
    else:
        raise RuntimeError(f"{name} still has incomplete rows; see {shard / 'details.log'}")
    elapsed = round(sum(item["seconds"] for item in attempts if item["seconds"] is not None))
    audit(stage_dir, elapsed)
    summary = stage_dir / "export" / "summary.json"
    result = load(summary)
    result.update(attempts=attempts, wall_seconds_is_complete=all(item["seconds"] is not None for item in attempts))
    revision.baseline.write(summary, result)


def check_devices():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 1 or not devices[0] or int(os.environ.get("SLURM_GPUS_ON_NODE", "1")) != 1:
        raise RuntimeError("Expected one Slurm-assigned CUDA device")
    import torch

    if torch.cuda.device_count() != 1 or "H20" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This suite is pinned to one visible H20")


def start_models(stack, args, ports):
    model = revision.runner.asset_path(services.MODEL)
    prm = services.start_service(
        stack, [sys.executable, services.__file__, "--serve-prm", "--prm-port", str(ports[1])],
        os.environ.copy(), args.output / "prm.log")
    engine = services.start_service(
        stack, [args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
                "--host", "127.0.0.1", "--port", str(ports[0]), "--served-model-name", services.MODEL[1],
                "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--generation-config", "vllm",
                "--gpu-memory-utilization", "0.60", "--max-model-len", "32768",
                "--enable-prefix-caching", "--max-num-seqs", "112", "--max-num-batched-tokens", "8192"],
        {key: val for key, val in os.environ.items() if key != "PYTHONPATH"}, args.output / "engine.log")
    services.wait_ready(prm, ports[1])
    services.wait_ready(engine, ports[0])


def coordinate(args):
    check_devices()
    revision.runner.ASSETS = (revision.runner.ASSETS[0], services.MODEL, revision.runner.ASSETS[2])
    if subprocess.check_output([args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))"],
                               text=True).strip() != "0.33.0":
        raise RuntimeError("Expected vLLM 0.33.0")
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "status.json"
    completed = []
    first_port = 22500 + 2 * (int(os.environ.get("SLURM_JOB_ID", "0")) % 10000)
    ports = (first_port, first_port + 1)
    revision.baseline.write(status, {"phase": "starting_services"})
    with ExitStack() as stack:
        start_models(stack, args, ports)
        for name in STAGES:
            for phase, sample, workers in (("smoke", 8, 8), ("full", None, args.workers)):
                root = args.output / phase
                if (root / name / "export" / "summary.json").is_file():
                    continue
                revision.baseline.write(status, {"phase": phase, "stage": name, "completed": completed})
                print(f"{phase} {name} started", flush=True)
                run_stage(root, name, sample, workers, ports, args.dataset_file)
            completed.append(name)
            print(f"full {name} audited", flush=True)
            if name == "answer-prm-sum":
                compare.compare(args.output / "full")
        revision.baseline.write(status, {"phase": "matched", "completed": completed})
        matched.run(args.output / "full")
    revision.baseline.write(status, {"phase": "verified", "completed": completed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--engine-python", required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    coordinate(args)


if __name__ == "__main__":
    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    main()

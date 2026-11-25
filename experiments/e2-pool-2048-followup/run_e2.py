"""Run the 2,048-token E2 candidate pools after the E4 server job."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from schedule import COHORTS, COUNTS, TARGET, eligible


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "direct-e4-pools-2048"))
from run_pools import (CAP, ENGINE, ENGINE_FLAGS, LANES, MODELS, PRM_PROMPT, WORKERS,
                       assigned, check_e4, frozen, released, sha, solve, start_service, wait_ready)


MANIFEST = EXPERIMENTS / "gsm8k-500-e4-direct/followup_questions.json"


def questions(data):
    groups = {name: [] for name in COHORTS}
    for q in frozen(data):
        if q["dataset"] in groups:
            groups[q["dataset"]].append({**q, "problem_id": q["source_problem_id"]})
    for dataset, rows in groups.items():
        if [q["problem_id"] for q in rows] != list(range(COUNTS[dataset])):
            raise ValueError(f"Unexpected {dataset} question set")
    return groups


def worker(args):
    from answer_fields import install

    install()
    label, gen, prm_port = LANES[args.lane]
    model = MODELS[label]
    runner = released.runner
    assets, _, prm = runner.ASSETS
    runner.ASSETS = (assets, model, prm)
    cfg = released.configuration("E2")
    cfg["client"]["generate_kwargs"]["max_tokens"] = CAP
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(cfg, raw)
    prompts = runner.upstream.get_prompts()
    evaluator = runner.upstream.get_evaluator("MATH")
    e2_args = SimpleNamespace(worker="fixed", lane=args.lane, gen_port=gen, prm_port=prm_port)
    cohort = assigned(questions(args.data)[args.dataset], args.lane)
    directory = args.output / "problems"
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "case": "E2", "dataset": args.dataset, "lane": args.lane, "model": list(model), "prm": list(prm),
        "decode": decode, "agent": cfg["agent"], "workers": WORKERS, "num_iterations": 10,
        "prm_prompt": PRM_PROMPT, "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": "0.33.0"},
        "upstream_commit": runner.UPSTREAM_COMMIT, "source_sha256": sha(Path(__file__).read_bytes()),
        "solve_sha256": sha((EXPERIMENTS / "llama-e4-online-ab/online_ab.py").read_bytes()),
        "question_manifest_sha256": sha(MANIFEST.read_bytes()),
        "ids": [q["problem_id"] for q in cohort],
    }
    file = args.output / "config.json"
    if file.exists() and released.read(file) != config:
        raise ValueError("Cannot resume E2 with different settings")
    released.write(file, config)

    def run_one(q):
        i = q["problem_id"]
        path = directory / f"{i:03d}.json"
        if path.exists():
            row = released.read(path)
        else:
            row = solve(q, e2_args, cfg, tokenizer, decode, evaluator, prompts, model=model, retain_traces=True)
            row.update(dataset=q["dataset"], gold=q["gold_answer_extracted"])
        if (row["problem_id"], row["dataset_unique_id"], row["dataset"], row["gold"]) != (
                i, q["dataset_unique_id"], q["dataset"], q["gold_answer_extracted"]):
            raise ValueError("Unexpected E2 checkpoint identity")
        check_e4(row, args.lane)
        if not path.exists():
            released.write(path, row)

    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
                runner.upstream, "create_llm_client", lambda _: released.THREAD.client):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs = [pool.submit(run_one, q) for q in cohort]
                for n, job in enumerate(as_completed(jobs), 1):
                    job.result()
                    if n % 25 == 0 or n == len(jobs):
                        print(f"E2 {args.dataset} GPU{args.lane} {n}/{len(jobs)}", file=sys.__stdout__, flush=True)


def coordinate(args):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(LANES) or len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three distinct GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {}
    with ExitStack() as stack:
        services = []
        for lane, device in zip(LANES, devices):
            if lane == 7:
                continue
            label, gen, prm = LANES[lane]
            model = MODELS[label]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            servers = {
                prm: [sys.executable, str(EXPERIMENTS / "llama-table1-825/llama_table1.py"),
                      "--serve-prm", "--prm-port", str(prm)],
                gen: [args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve",
                      str(released.runner.asset_path(model)), "--host", "127.0.0.1", "--port", str(gen),
                      "--served-model-name", model[1], *ENGINE_FLAGS],
            }
            for port, command in servers.items():
                service_env = env if port == prm else {k: v for k,v in env.items() if k != "PYTHONPATH"}
                child = start_service(stack, command, service_env, args.output / f"gpu-{lane}-{port}.log")
                services.append((child, port))
        for child, port in services:
            wait_ready(child, port)
        for dataset in COHORTS:
            now = datetime.now(TARGET.tzinfo)
            record = {"decision_time": now.isoformat()}
            if not eligible(dataset, now) or (dataset == "amo" and runs["aime24"]["status"] != "complete"):
                runs[dataset] = {**record, "status": "deferred_by_window"}
                continue
            target = args.output / dataset
            target.mkdir(exist_ok=True)
            released.write(args.output / "status.json", {"phase": dataset, "target": TARGET.isoformat()})
            started = time.monotonic()
            codes = dict.fromkeys(LANES, -1)
            for attempt in range(2):
                with ExitStack() as workers:
                    children = {
                        lane: start_service(workers, [
                            sys.executable, "-B", str(Path(__file__)), "--lane", str(lane), "--dataset", dataset,
                            "--data", str(args.data), "--output", str(target / f"gpu-{lane}"),
                        ], {**os.environ, "CUDA_VISIBLE_DEVICES": device}, target / f"gpu-{lane}-worker.log")
                        for lane, device in zip(LANES, devices) if codes[lane]
                    }
                    codes.update({lane: child.wait() for lane, child in children.items()})
                if not any(codes.values()):
                    break
            if any(codes.values()):
                raise RuntimeError(f"E2 {dataset}: worker exit codes {codes}")
            run = {**record, "status": "complete", "worker_exit_codes": codes, "wall_seconds": time.monotonic() - started}
            released.write(target / "generation.json", run)
            runs[dataset] = run
    released.write(args.output / "generation.json", {"target": TARGET.isoformat(), "cohorts": runs})
    released.write(args.output / "status.json", {"phase": "complete", "cohorts": runs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    parser.add_argument("--dataset", choices=COHORTS)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.preflight:
        print({k: len(v) for k,v in questions(args.data).items()}, TARGET.isoformat(), flush=True)
    elif args.lane is not None:
        worker(args)
    else:
        coordinate(args)

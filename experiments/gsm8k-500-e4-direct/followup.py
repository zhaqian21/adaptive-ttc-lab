"""Run single-call direct generation on the five previously evaluated datasets."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from evaluate import (
    DIRECT_KEYS, DIRECT_WORKERS, ENGINE, ENGINE_FLAGS, HERE, LANES, MODELS, assigned, direct,
    released, sha, start_service, validate_direct, wait_ready,
)
from benchmarks import DATASETS, load as prior_load
from followup_benchmarks import load_followup_benchmark


COUNTS = {"math": 500, "amc23": 40, "aime24": 30, "amo": 50, "aime26": 30}
PLAN = HERE / "followup_questions.json"


def load_questions(data):
    questions, sources = [], {}
    runner = released.runner
    repo = ["git", "-C", str(runner.ROOT / "upstream")]
    if (subprocess.check_output([*repo, "rev-parse", "HEAD"], text=True).strip() != runner.UPSTREAM_COMMIT
            or subprocess.check_output([*repo, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected clean pinned upstream")
    for dataset, count in COUNTS.items():
        if dataset == "amc23":
            group, spec = load_followup_benchmark("llama-amc23", released)
        elif dataset == "aime26":
            group = [q for q in prior_load(data) if q["dataset"] == dataset]
            spec = DATASETS[dataset]
        else:
            group, spec = released.load_benchmark(
                SimpleNamespace(dataset=dataset, dataset_manifest=data / "manifest.json"))
            if dataset == "math":
                _, repo_name, revision = runner.ASSETS[0]
                spec = {**spec, "repo": repo_name, "revision": revision}
        if len(group) != count or len({q["dataset_unique_id"] for q in group}) != count:
            raise ValueError(f"{dataset}: unexpected dataset identity")
        sources[dataset] = {**spec, "samples": count}
        for index, question in enumerate(group):
            questions.append({**question, "problem_id": len(questions), "dataset": dataset,
                              "source_problem_id": index})
    plan = {
        "sources": sources,
        "questions": [{**{k: q[k] for k in ("problem_id", "source_problem_id", "dataset",
                                            "dataset_unique_id", "gold_answer_extracted")},
                        "question_sha256": sha(q["problem"].encode("utf-8"))} for q in questions],
    }
    return questions, plan


def frozen(data):
    questions, plan = load_questions(data)
    if plan != released.read(PLAN):
        raise ValueError("Wrong locked question manifest")
    return questions


def worker(args):
    qs = assigned(frozen(args.data), args.lane)
    label, gen_port, _ = LANES[args.lane]
    model = MODELS[label]
    runner = released.runner
    datasets, _, prm = runner.ASSETS
    runner.ASSETS = (datasets, model, prm)
    cfg = released.configuration("E4")
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(cfg, raw)
    evaluator = runner.upstream.get_evaluator("MATH")
    prompts = runner.upstream.get_prompts()
    run_args = SimpleNamespace(lane=args.lane, gen_port=gen_port)
    path = args.output / "direct"
    path.mkdir(parents=True, exist_ok=True)
    settings = {
        "lane": args.lane, "model": list(model), "decode": decode, "workers": DIRECT_WORKERS,
        "direct_prompts": {key: prompts[key] for key in DIRECT_KEYS}, "upstream_commit": runner.UPSTREAM_COMMIT,
        "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": "0.33.0"},
        "plan_sha256": sha(PLAN.read_bytes()), "source_sha256": sha(Path(__file__).read_bytes()),
        "shared_direct_source_sha256": sha((HERE / "evaluate.py").read_bytes()),
        "ids": [q["problem_id"] for q in qs],
    }
    config_path = args.output / "config.json"
    if config_path.exists() and released.read(config_path) != settings:
        raise ValueError("Cannot resume with changed settings")
    released.write(config_path, settings)

    def evaluate(q):
        output = path / f"{q['problem_id']:03d}.json"
        if output.exists():
            row = released.read(output)
        else:
            row = direct(q, run_args, cfg["client"], tokenizer, decode, evaluator, prompts, model)
            row["source_problem_id"] = q["source_problem_id"]
        validate_direct(row, args.lane)
        if any(row[k] != q[k] for k in ("problem_id", "source_problem_id", "dataset", "dataset_unique_id")):
            raise ValueError("Incorrect question identity")
        if row["gold_answer"] != q["gold_answer_extracted"]:
            raise ValueError("Correct answer reference changed")
        if not output.exists():
            released.write(output, row)

    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), ThreadPoolExecutor(max_workers=DIRECT_WORKERS) as pool:
            jobs = [pool.submit(evaluate, q) for q in qs]
            for count, future in enumerate(as_completed(jobs), 1):
                future.result()
                if count % 50 == 0 or count == len(jobs):
                    print(f"GPU {args.lane} {label}: {count}/{len(jobs)} complete", file=sys.__stdout__, flush=True)


def coordinate(args):
    frozen(args.data)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three separate GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Unexpected vLLM version {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ExitStack() as stack:
        services, workers = [], []
        for lane, device in zip(LANES, devices):
            if lane == 7:
                continue
            label, port, _ = LANES[lane]
            model = MODELS[label]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            child = start_service(stack, [
                args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve",
                str(released.runner.asset_path(model)), "--host", "127.0.0.1", "--port", str(port),
                "--served-model-name", model[1], *ENGINE_FLAGS,
            ], {k: v for k, v in env.items() if k != "PYTHONPATH"}, args.output / f"gpu-{lane}-engine.log")
            services.append((child, port))
        for child, port in services:
            wait_ready(child, port)
        for lane, device in zip(LANES, devices):
            entry = start_service(stack, [
                sys.executable, "-B", str(Path(__file__)), "--lane", str(lane), "--data", str(args.data),
                "--output", str(args.output / f"gpu-{lane}"),
            ], {**os.environ, "CUDA_VISIBLE_DEVICES": device}, args.output / f"gpu-{lane}-worker.log")
            workers.append((lane, entry))
        codes = {str(lane): process.wait() for lane, process in workers}
        if any(codes.values()):
            raise RuntimeError(f"Direct workers failed: {codes}")
    released.write(args.output / "generation.json", {"wall_seconds": time.monotonic() - started, "exit_codes": codes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    parser.add_argument("--freeze-plan", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.freeze_plan:
        _, plan = load_questions(args.data)
        released.write(PLAN, plan)
    elif args.lane is not None:
        worker(args)
    else:
        coordinate(args)

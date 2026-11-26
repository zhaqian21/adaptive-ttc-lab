"""Run the Qwen E2 dynamic-budget comparison on three H20s."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from importlib.metadata import version
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "e2-pool-2048-followup"))
from run_e2 import MANIFEST, questions
from run_pools import (CAP, CONTEXT, DIRECT, ENGINE, ENGINE_FLAGS, MODELS, PRM_PROMPT,
                       WORKERS, released, sha, solve, start_service, wait_ready)
from online_ab import COST, valid_stages
from protocol import DRAWS, key, majority


LANES = {0: ("e2_fixed10", 22500, 22502), 1: ("e2_dynamic3_to_5", 22501, 22503), 7: ("direct_vote5", 22504, None)}
QWEN = MODELS["qwen"]
SMOKE_IDS = (0, 1, 2, 3)


def agree(candidates):
    answers = [key(candidate["predicted_answer"]) for candidate in candidates]
    return len(answers) == 3 and bool(answers[0]) and all(x == answers[0] for x in answers)


def validate(row, question, lane):
    method, _, prm = LANES[lane]
    if (row["problem_id"], row["dataset_unique_id"], row["gold"], row["lane"], row["method"]) != (
            question["problem_id"], question["dataset_unique_id"], question["gold_answer_extracted"], lane, method):
        raise ValueError("Unexpected question identity or method")
    if row["seconds"] < 0 or row["generation_without_usage"]:
        raise ValueError("Missing usage or elapsed time")
    if prm is None:
        draws = row["outputs"]
        if (len(draws) != DRAWS or row["generation_requests"] != DRAWS or row["prm_requests"]
                or [d["seed"] for d in draws] != [question["problem_id"] + i * 1_000_000 for i in range(DRAWS)]
                or row["answer"] != draws[majority([d["full"] for d in draws])]["full"]):
            raise ValueError("Unexpected Direct vote or usage")
        return

    candidates = row["candidates"]
    dynamic = method == "e2_dynamic3_to_5"
    stopped = dynamic and agree(candidates[:3])
    limit = 5 if dynamic else 10
    if row["stopped_early"] != stopped or [c["iteration"] for c in candidates] != list(range(1, (3 if stopped else limit) + 1)):
        raise ValueError("Unexpected adaptive path")
    valid_stages(row["attempts"][-1]["stages"], stopped, limit, 3)
    for cost in COST:
        if row[cost] != sum(stage[cost] for attempt in row["attempts"] for stage in attempt["stages"]):
            raise ValueError(f"Unexpected usage for {cost}")
    selected = 1
    if not stopped:
        rewards = [c["mean_reward"] for c in candidates]
        selected = rewards.index(max(rewards)) + 1 if any(rewards) else limit
        if any(not c["prm_input"] or c["prm_score"]["mean_reward"] != c["mean_reward"] for c in candidates):
            raise ValueError("Missing PRM traces")
    if row["answer"]["selected_iteration"] != selected:
        raise ValueError("Unexpected original PRM choice")


def worker(args):
    from answer_fields import install

    install()
    method, gen, prm = LANES[args.lane]
    runner = released.runner
    assets, _, reward = runner.ASSETS
    runner.ASSETS = (assets, QWEN, reward)
    cfg = released.configuration("E2")
    cfg["client"]["generate_kwargs"]["max_tokens"] = CAP
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(cfg, raw)
    prompts = runner.upstream.get_prompts()
    evaluator = runner.upstream.get_evaluator("MATH")
    proxy = SimpleNamespace(port=gen, model=QWEN[1], context=CONTEXT)
    dynamic = method == "e2_dynamic3_to_5"
    solve_args = SimpleNamespace(worker="early" if dynamic else "fixed", lane=args.lane, gen_port=gen, prm_port=prm)
    cohort = questions(args.data)["math"]
    if args.smoke:
        cohort = [q for q in cohort if q["problem_id"] in SMOKE_IDS]
    paths = args.output / "problems"
    paths.mkdir(parents=True, exist_ok=True)
    config = {
        "method": method, "lane": args.lane, "model": list(QWEN), "prm": list(reward) if prm is not None else None,
        "workers": WORKERS, "decode": decode, "agent": cfg["agent"], "prm_prompt": PRM_PROMPT,
        "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": "0.33.0"},
        "upstream_commit": runner.UPSTREAM_COMMIT, "question_manifest_sha256": sha(MANIFEST.read_bytes()),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "solve_sha256": sha((EXPERIMENTS / "llama-e4-online-ab/online_ab.py").read_bytes()),
        "ids": [q["problem_id"] for q in cohort],
        "direct_prompts": {k: prompts[k] for k in DIRECT["DIRECT_KEYS"]} if prm is None else None,
    }
    config_path = args.output / "config.json"
    if config_path.exists() and released.read(config_path) != config:
        raise ValueError("Cannot resume the experiment with different settings")
    released.write(config_path, config)
    existing = sum((paths / f"{q['problem_id']:03d}.json").is_file() for q in cohort)

    def run_one(q):
        path = paths / f"{q['problem_id']:03d}.json"
        if path.exists():
            validate(released.read(path), q, args.lane)
            return
        if prm is None:
            started = time.monotonic()
            draws = [DIRECT["one"]({**q, "seed": q["problem_id"]}, proxy, cfg, tokenizer, decode, evaluator, prompts, i)
                     for i in range(DRAWS)]
            result = {
                "problem_id": q["problem_id"], "dataset_unique_id": q["dataset_unique_id"], "lane": args.lane,
                "outputs": draws, "answer": draws[majority([d["full"] for d in draws])]["full"],
                "seconds": time.monotonic() - started, **{cost: sum(d[cost] for d in draws) for cost in COST},
            }
        else:
            result = solve(q, solve_args, cfg, tokenizer, decode, evaluator, prompts, model=QWEN,
                           retain_traces=True, candidate_limit=5 if dynamic else 10, agreement_at=3,
                           agreement_rule=agree, original_prm_only=True)
        result.update(method=method, gold=q["gold_answer_extracted"])
        validate(result, q, args.lane)
        released.write(path, result)

    started = time.monotonic()
    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
                runner.upstream, "create_llm_client", lambda _: released.THREAD.client):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                jobs = [pool.submit(run_one, q) for q in cohort]
                for n, job in enumerate(as_completed(jobs), 1):
                    job.result()
                    if n % 25 == 0 or n == len(jobs):
                        print(f"{method} GPU{args.lane} {n}/{len(jobs)}", file=sys.__stdout__, flush=True)
    timing = args.output / "timing.json"
    if not (existing == len(cohort) and timing.exists()):
        released.write(timing, {"wall_seconds": time.monotonic() - started, "resumed_questions": existing})


def coordinate(args):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(LANES) or len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three distinct GPUs")
    engine_version = subprocess.check_output(
        [args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))"],
        text=True).strip()
    if engine_version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {engine_version}")
    args.output.mkdir(parents=True, exist_ok=True)
    released.write(args.output / "status.json", {"phase": "starting_services"})
    with ExitStack() as services:
        ready = []
        for (lane, (_, gen, prm)), gpu in zip(LANES.items(), devices):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            engine_env = {k: v for k, v in env.items() if k != "PYTHONPATH"}
            gen_cmd = [args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve", str(released.runner.asset_path(QWEN)),
                       "--host", "127.0.0.1", "--port", str(gen), "--served-model-name", QWEN[1], *ENGINE_FLAGS]
            ready.append((start_service(services, gen_cmd, engine_env, args.output / f"gpu-{lane}-generator.log"), gen))
            if prm is not None:
                prm_cmd = [sys.executable, str(EXPERIMENTS / "llama-table1-825/llama_table1.py"),
                           "--serve-prm", "--prm-port", str(prm)]
                ready.append((start_service(services, prm_cmd, env, args.output / f"gpu-{lane}-prm.log"), prm))
        for process, port in ready:
            wait_ready(process, port)
        for phase in ("smoke", "formal"):
            released.write(args.output / "status.json", {"phase": phase})
            with ExitStack() as workers:
                children = {}
                for (lane, (method, _, _)), gpu in zip(LANES.items(), devices):
                    output = args.output / phase / method
                    output.mkdir(parents=True, exist_ok=True)
                    cmd = [sys.executable, "-B", str(Path(__file__)), "--lane", str(lane), "--data", str(args.data),
                           "--output", str(output)]
                    if phase == "smoke":
                        cmd.append("--smoke")
                    children[method] = start_service(workers, cmd, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                                     output / "worker.log")
                codes = {method: p.wait() for method, p in children.items()}
                if any(codes.values()):
                    raise RuntimeError(f"{phase}: {codes}")
    released.write(args.output / "status.json", {"phase": "generation-complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=LANES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.preflight:
        print(f"{len(questions(args.data)['math'])} frozen MATH-500 questions")
    elif args.lane is not None:
        worker(args)
    else:
        coordinate(args)

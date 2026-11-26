"""Collect six-dataset Direct and original E4 pools with per-request cap 2,048."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import os
import runpy
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "math500-direct-continuation"
sys.path.insert(0, str(PRIOR))
DIRECT = runpy.run_path(str(PRIOR / "run.py"), run_name="prior_direct_runner")
from evaluate import (CONTEXT, DIRECT_KEYS, DIRECT_WORKERS, ENGINE, ENGINE_FLAGS, LANES, MODELS,
                      WORKERS, assigned, released, sha, start_service, verify_row, wait_ready)
from followup import frozen
from online_ab import PRM_PROMPT, solve
from protocol import DRAWS, LONG
from sample import load as load_gsm, plan as gsm_plan


PHASES = ("direct5", "e4")
CAP = LONG
PLAN = HERE / "question_plan.json"


def questions(data):
    prior = frozen(data)
    gs = load_gsm(data)
    if gsm_plan(gs) != released.read(HERE.parent / "gsm8k-500-e4-direct/questions.json"):
        raise ValueError("The GSM8K asset and prior sample differ")
    catalog = {(q["dataset"], q["dataset_unique_id"]): q for q in prior + gs}
    plan = released.read(PLAN)
    if (len(plan) != 1150 or len(catalog) != 1150
            or [q["problem_id"] for q in plan] != list(range(1150))):
        raise ValueError("Unexpected question catalog")
    result = []
    for item in plan:
        q = catalog[(item["dataset"], item["dataset_unique_id"])]
        if (sha(q["problem"].encode("utf-8")) != item["question_sha256"]
                or q["gold_answer_extracted"] != item["gold_answer_extracted"]
                or q["problem_id"] != item["direct_seed"]):
            raise ValueError(f"Question differs from the prior manifest: {item['problem_id']}")
        result.append({**q, **item})
    return result


def check_e4(result, lane):
    verify_row(result, "fixed", expected_lane=lane)
    for candidate in result["candidates"]:
        if (sha(candidate["answer_unstructured"].encode("utf-8")) != candidate["completion_sha256"]
                or not candidate["prm_input"]
                or candidate["prm_score"]["mean_reward"] != candidate["mean_reward"]):
            raise ValueError("Missing or changed E4 rescoring inputs")


def worker(args):
    from answer_fields import install

    install()
    model_name, gen, prm_port = LANES[args.lane]
    model = MODELS[model_name]
    runner = released.runner
    datasets, _, prm = runner.ASSETS
    runner.ASSETS = (datasets, model, prm)
    cfg = released.configuration("E4")
    cfg["client"]["generate_kwargs"]["max_tokens"] = CAP
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(cfg, raw)
    prompts = runner.upstream.get_prompts()
    evaluator = runner.upstream.get_evaluator("MATH")
    model_args = SimpleNamespace(port=gen, model=model[1], context=CONTEXT)
    e4_args = SimpleNamespace(worker="fixed", lane=args.lane, gen_port=gen, prm_port=prm_port)
    cohort = assigned(questions(args.data), args.lane)
    max_workers = DIRECT_WORKERS if args.phase == "direct5" else WORKERS
    directory = args.output / "problems"
    directory.mkdir(parents=True, exist_ok=True)
    settings = {
        "phase": args.phase, "lane": args.lane, "model": list(model), "prm": list(prm),
        "model_name": model_name, "decode": decode, "agent": cfg["agent"], "workers": max_workers,
        "upstream_commit": runner.UPSTREAM_COMMIT, "prm_prompt": PRM_PROMPT,
        "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": "0.33.0"},
        "direct_prompts": {k: prompts[k] for k in DIRECT_KEYS}, "question_plan_sha256": sha(PLAN.read_bytes()),
        "source_sha256": sha(Path(__file__).read_bytes()), "direct_source_sha256": sha((PRIOR / "run.py").read_bytes()),
        "e4_source_sha256": sha((HERE.parent / "llama-e4-online-ab/online_ab.py").read_bytes()),
        "ids": [q["problem_id"] for q in cohort],
    }
    settings_path = args.output / "config.json"
    if settings_path.exists() and released.read(settings_path) != settings:
        raise ValueError("Cannot resume a shard with a different generation profile")
    released.write(settings_path, settings)

    def run_one(q):
        identity = {k: q[k] for k in ("problem_id", "dataset", "dataset_unique_id")}
        identity["gold"] = q["gold_answer_extracted"]
        target = directory / f"{q['problem_id']:04d}.json"
        if target.exists():
            saved = released.read(target)
            if any(saved[k] != v for k, v in identity.items()) or saved["lane"] != args.lane:
                raise ValueError("Existing checkpoint identity differs")
            if args.phase == "e4":
                check_e4(saved["result"], args.lane)
            elif len(saved["outputs"]) != DRAWS:
                raise ValueError("Incomplete Direct checkpoint")
            return
        if args.phase == "direct5":
            result = {"outputs": [
                DIRECT["one"]({**q, "seed": q["direct_seed"]}, model_args, cfg, tokenizer, decode, evaluator, prompts, i)
                for i in range(DRAWS)
            ]}
        else:
            episode = solve({**q, "problem_id": q["e4_seeds"][model_name]}, e4_args, cfg, tokenizer,
                            decode, evaluator, prompts, model=model, retain_traces=True)
            check_e4(episode, args.lane)
            result = {"result": episode}
        released.write(target, {**identity, "lane": args.lane, **result})

    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
                runner.upstream, "create_llm_client", lambda _: released.THREAD.client):
            with ThreadPoolExecutor(max_workers=max_workers) as threads:
                jobs = [threads.submit(run_one, q) for q in cohort]
                for count, future in enumerate(as_completed(jobs), 1):
                    future.result()
                    if count % 25 == 0 or count == len(jobs):
                        print(f"{args.phase} GPU {args.lane} {model_name}: {count}/{len(jobs)}",
                              file=sys.__stdout__, flush=True)


def coordinate(args):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(LANES) or len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three distinct allocated GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Unexpected vLLM version {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    released.write(args.output / "status.json", {"phase": "starting_services"})
    with ExitStack() as stack:
        services = []
        for lane, device in zip(LANES, devices):
            if lane == 7:
                continue
            label, gen, prm = LANES[lane]
            model = MODELS[label]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            servers = {
                prm: [sys.executable, str(HERE.parent / "llama-table1-825/llama_table1.py"),
                      "--serve-prm", "--prm-port", str(prm)],
                gen: [args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve",
                      str(released.runner.asset_path(model)), "--host", "127.0.0.1", "--port", str(gen),
                      "--served-model-name", model[1], *ENGINE_FLAGS],
            }
            for port, command in servers.items():
                service_env = env if port == prm else {k: v for k, v in env.items() if k != "PYTHONPATH"}
                child = start_service(stack, command, service_env, args.output / f"gpu-{lane}-{port}.log")
                services.append((child, port))
        for child, port in services:
            wait_ready(child, port)
        for phase in PHASES:
            target = args.output / phase
            target.mkdir(exist_ok=True)
            released.write(args.output / "status.json", {"phase": phase})
            started = time.monotonic()
            codes = dict.fromkeys(LANES, -1)
            for attempt in range(2):
                with ExitStack() as workers:
                    children = {
                        lane: start_service(workers, [
                            sys.executable, "-B", str(Path(__file__)), "--lane", str(lane), "--phase", phase,
                            "--data", str(args.data), "--output", str(target / f"gpu-{lane}"),
                        ], {**os.environ, "CUDA_VISIBLE_DEVICES": device}, target / f"gpu-{lane}-worker.log")
                        for lane, device in zip(LANES, devices) if codes[lane]
                    }
                    codes.update({lane: child.wait() for lane, child in children.items()})
                if not any(codes.values()):
                    break
            if any(codes.values()):
                raise RuntimeError(f"{phase}: worker exit codes {codes}")
            released.write(target / "generation.json", {"worker_exit_codes": codes,
                                                        "wall_seconds": time.monotonic() - started})
    released.write(args.output / "status.json", {"phase": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-python")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    worker(args) if args.lane is not None else coordinate(args)

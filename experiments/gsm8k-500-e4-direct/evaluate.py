"""Generate the 300 extra E4 pools and 500 single-call direct answers per model."""
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
from unittest.mock import patch

from sample import HERE, PREVIOUS, REUSED, load, plan, sha

from run import (
    CONTEXT, ENGINE, ENGINE_FLAGS, MODELS, WORKERS, released, solve,
    start_service, verify_row, wait_ready,
)
from online_ab import MeasuredSession


LANES = {0: ("qwen", 22500, 22502), 1: ("llama", 22501, 22503), 7: ("llama", 18427, 18437)}
DIRECT_WORKERS = 48
DIRECT_KEYS = ("direct_unstructured_final_answer_system_prompt", "direct_unstructured_final_answer_user_prompt")


def questions(args):
    qs = load(args.data)
    if plan(qs) != released.read(HERE / "questions.json"):
        raise ValueError("Unexpected frozen question plan")
    runner = released.runner
    git = ["git", "-C", str(runner.ROOT / "upstream")]
    if (subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != runner.UPSTREAM_COMMIT
            or subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected the pinned clean upstream")
    return qs


def assigned(qs, lane):
    if lane == 0:
        return qs
    return [q for q in qs if q["problem_id"] % 2 == (lane == 7)]


def direct(q, args, client_config, tokenizer, decode, evaluator, prompts, model):
    """Use the judged unstructured prompt from upstream Direct, in one call."""
    started = time.monotonic()
    index = q["problem_id"]
    messages = [
        {"role": "system", "content": prompts[DIRECT_KEYS[0]]},
        {"role": "user", "content": prompts[DIRECT_KEYS[1]].format(problem=q["problem"])},
    ]
    with MeasuredSession() as session:
        session.stage("direct")
        client = released.TransformerWrapper(released.runner.upstream.DictConfig(client_config))
        client._initialized = True
        client.tokenizer = tokenizer
        client.model = released.GenerationProxy(
            SimpleNamespace(port=args.gen_port, context=CONTEXT, model=model[1]), decode, index, [], session)
        response = client.generate(messages)
        answer = evaluator._extract_answer_from_model_completion(response.completion)
        correct = evaluator._check_answers_equiv(answer, q["gold_answer_extracted"])
        costs = session.stages[0]
    return {
        "problem_id": index, "dataset_unique_id": q["dataset_unique_id"],
        "dataset": q["dataset"], "gold_answer": q["gold_answer_extracted"],
        "lane": args.lane, "seed": index, "predicted_answer": answer,
        "is_correct": correct is True, "completion_sha256": sha(response.completion.encode("utf-8")),
        "seconds": time.monotonic() - started, **costs,
    }


def validate_direct(row, lane):
    if (row["lane"] != lane or row["seed"] != row["problem_id"] or row["stage"] != "direct"
            or type(row["is_correct"]) is not bool or row["prm_requests"] != 0
            or row["generation_requests"] - row["generation_without_usage"] != 1
            or min(row["input_tokens"], row["output_tokens"]) < 1):
        raise ValueError(f"Invalid direct-only ledger for question {row['problem_id']}")


def evaluate(args):
    from answer_fields import install

    install()
    qs = assigned(questions(args), args.lane)
    label, gen_port, prm_port = LANES[args.lane]
    model = MODELS[label]
    runner = released.runner
    datasets, _, prm = runner.ASSETS
    runner.ASSETS = (datasets, model, prm)
    cfg = released.configuration("E4")
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(cfg, raw)
    prompts = runner.upstream.get_prompts()
    evaluator = runner.upstream.get_evaluator("MATH")
    run_args = SimpleNamespace(worker="fixed", lane=args.lane, gen_port=gen_port, prm_port=prm_port)
    settings = {
        "lane": args.lane, "model": list(model), "prm": list(prm),
        "upstream_commit": runner.UPSTREAM_COMMIT, "decode": decode, "agent": cfg["agent"],
        "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": "0.33.0"},
        "e4_workers": WORKERS, "direct_workers": DIRECT_WORKERS,
        "direct_prompts": {k: prompts[k] for k in DIRECT_KEYS},
        "direct_ids": [q["problem_id"] for q in qs], "e4_ids": [q["problem_id"] for q in qs if q["problem_id"] >= REUSED],
        "manifest_sha256": sha((HERE / "questions.json").read_bytes()),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "e4_solve_sha256": sha((HERE.parent / "llama-e4-online-ab/online_ab.py").read_bytes()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output / "config.json"
    if cfg_path.exists() and released.read(cfg_path) != settings:
        raise ValueError("Cannot resume with changed settings")
    released.write(cfg_path, settings)

    def run_phase(name, batch, concurrency):
        path = args.output / name
        path.mkdir(exist_ok=True)

        def run_one(q):
            path_q = path / f"{q['problem_id']:03d}.json"
            if path_q.exists():
                row = released.read(path_q)
            elif name == "direct":
                row = direct(q, run_args, cfg["client"], tokenizer, decode, evaluator, prompts, model)
            else:
                row = solve(q, run_args, cfg, tokenizer, decode, evaluator, prompts, model=model)
                row.update(dataset=q["dataset"], gold_answer=q["gold_answer_extracted"])
            if (row["problem_id"], row["dataset_unique_id"], row["gold_answer"]) != (
                    q["problem_id"], q["dataset_unique_id"], q["gold_answer_extracted"]):
                raise ValueError("Result does not match assigned question")
            if name == "direct":
                validate_direct(row, args.lane)
            else:
                verify_row(row, "fixed", expected_lane=args.lane)
            if not path_q.exists():
                released.write(path_q, row)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            jobs = [pool.submit(run_one, q) for q in batch]
            for i, job in enumerate(as_completed(jobs), 1):
                job.result()
                if i % 25 == 0 or i == len(jobs):
                    print(f"GPU {args.lane} {label} {name}: {i}/{len(jobs)} complete", file=sys.__stdout__, flush=True)

    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
                runner.upstream, "create_llm_client", lambda _: released.THREAD.client):
            run_phase("direct", qs, DIRECT_WORKERS)
            run_phase("e4", [q for q in qs if q["problem_id"] >= REUSED], WORKERS)


def coordinate(args):
    questions(args)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three distinct physical GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ExitStack() as stack:
        services, workers = [], []
        for lane, device in zip(LANES, devices):
            label, gen_port, prm_port = LANES[lane]
            if lane == 7:
                continue
            model = MODELS[label]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            prm = start_service(stack, [
                sys.executable, str(HERE.parent / "llama-table1-825/llama_table1.py"),
                "--serve-prm", "--prm-port", str(prm_port),
            ], env, args.output / f"gpu-{lane}-prm.log")
            llm = start_service(stack, [
                args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve",
                str(released.runner.asset_path(model)), "--host", "127.0.0.1", "--port", str(gen_port),
                "--served-model-name", model[1], *ENGINE_FLAGS,
            ], {k: v for k, v in env.items() if k != "PYTHONPATH"}, args.output / f"gpu-{lane}-engine.log")
            services.extend(((prm, prm_port), (llm, gen_port)))
        for child, port in services:
            wait_ready(child, port)
        for lane, device in zip(LANES, devices):
            worker = start_service(stack, [
                sys.executable, "-B", str(Path(__file__)), "--lane", str(lane), "--data", str(args.data),
                "--output", str(args.output / f"gpu-{lane}"),
            ], {**os.environ, "CUDA_VISIBLE_DEVICES": device}, args.output / f"gpu-{lane}-worker.log")
            workers.append((lane, worker))
        codes = {str(lane): worker.wait() for lane, worker in workers}
        if any(codes.values()):
            raise RuntimeError(f"Worker failure: {codes}")
    released.write(args.output / "generation.json", {"wall_seconds": time.monotonic() - started, "worker_exit_codes": codes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    evaluate(args) if args.lane is not None else coordinate(args)


if __name__ == "__main__":
    main()

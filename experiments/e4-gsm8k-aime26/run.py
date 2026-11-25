"""Build one ordered 10-candidate E4 pool per model and problem on dual H20s."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import load, manifest, rows_by_dataset, sha


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llama-e4-online-ab"))
from online_ab import CONTEXT, MODEL, PORTS, WORKERS, released, solve, start_service, wait_ready
from compare_online import verify_row
from followup_benchmarks import QWEN


MODELS = {"llama": MODEL, "qwen": QWEN}
ENGINE = {
    "dtype": "bfloat16", "tensor-parallel-size": 1, "generation-config": "vllm",
    "gpu-memory-utilization": 0.60, "max-model-len": CONTEXT,
    "max-num-seqs": 112, "max-num-batched-tokens": 8192,
}
ENGINE_FLAGS = [
    part for key, value in ENGINE.items() for part in ("--" + key, str(value))
] + ["--enable-prefix-caching"]


def inputs(args):
    questions = load(args.data)
    expected = manifest(questions)
    if released.read(HERE / "questions.json") != expected:
        raise ValueError("Candidate generation requires the frozen question manifest")
    runner = released.runner
    upstream = ["git", "-C", str(runner.ROOT / "upstream")]
    if (subprocess.check_output([*upstream, "rev-parse", "HEAD"], text=True).strip() != runner.UPSTREAM_COMMIT
            or subprocess.check_output([*upstream, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected the clean pinned upstream")
    evaluator = runner.upstream.get_evaluator("AIME2024")
    for q in questions:
        gold = q["gold_answer_extracted"]
        extracted = evaluator._extract_answer_from_model_completion(r"\boxed{" + gold + "}")
        if not extracted or not evaluator._check_answers_equiv(extracted, gold):
            raise ValueError(f"Gold answer failed grading control: {q['dataset']}/{q['dataset_unique_id']}")
    if not evaluator._check_answers_equiv("007", "7") or evaluator._check_answers_equiv("7", "17"):
        raise ValueError("Integer-equivalence grading controls failed")
    return questions, evaluator


def worker(args):
    from answer_fields import install

    install()
    questions, evaluator = inputs(args)
    model = MODELS[args.model]
    runner = released.runner
    datasets, _, prm = runner.ASSETS
    runner.ASSETS = (datasets, model, prm)
    config = released.configuration("E4")
    raw = released.AutoTokenizer.from_pretrained(config["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(config, raw)
    prompts = runner.upstream.get_prompts()
    lane = tuple(MODELS).index(args.model)
    worker_args = SimpleNamespace(worker="fixed", lane=lane, gen_port=PORTS[lane], prm_port=PORTS[lane + 2])
    problem_dir = args.output / "problems"
    problem_dir.mkdir(parents=True, exist_ok=True)
    setup = {
        "model": list(model), "prm": list(prm), "lane": lane, "workers": WORKERS,
        "upstream_commit": runner.UPSTREAM_COMMIT, "decode": decode, "agent": config["agent"],
        "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": args.vllm_version},
        "num_iterations": 10, "seed_rule": "problem_id + 1000000 * retry + 500 * prior_calls",
        "manifest_sha256": sha((HERE / "questions.json").read_bytes()),
        "source_sha256": sha(Path(__file__).read_bytes()),
        "online_ab_sha256": sha((HERE.parent / "llama-e4-online-ab/online_ab.py").read_bytes()),
    }
    settings = args.output / "config.json"
    if settings.exists() and released.read(settings) != setup:
        raise ValueError("Resume requires identical settings")
    released.write(settings, setup)

    def evaluate(question):
        index = question["problem_id"]
        path = problem_dir / f"{index:03d}.json"
        if path.exists():
            row = released.read(path)
        else:
            row = solve(question, worker_args, config, tokenizer, decode, evaluator, prompts, model=model)
            row.update(dataset=question["dataset"], gold_answer=question["gold_answer_extracted"])
            verify_row(row, "fixed", expected_lane=lane)
            released.write(path, row)
        if (row["problem_id"], row["dataset"], row["dataset_unique_id"], row["gold_answer"]) != (
                index, question["dataset"], question["dataset_unique_id"], question["gold_answer_extracted"]):
            raise ValueError(f"Checkpoint identity mismatch: question {index}")

    groups = rows_by_dataset(questions)
    smoke = [q for group in groups.values() for q in group[:2]]
    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(
                runner.upstream, "create_llm_client", lambda _: released.THREAD.client):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                list(pool.map(evaluate, smoke))
                print(f"{args.model}: verified 2 questions per dataset", file=sys.__stdout__, flush=True)
                for dataset, group in groups.items():
                    futures = [pool.submit(evaluate, q) for q in group[2:]]
                    for count, future in enumerate(as_completed(futures), start=3):
                        future.result()
                        if count % 8 == 0 or count == len(group):
                            print(f"{args.model} {dataset}: {count}/{len(group)} complete",
                                  file=sys.__stdout__, flush=True)


def coordinate(args):
    inputs(args)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or not all(devices) or len(set(devices)) != 2:
        raise ValueError("Two distinct GPUs must be allocated")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ExitStack() as stack:
        services, workers = [], []
        for lane, (label, model) in enumerate(MODELS.items()):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[lane]}
            prm = start_service(stack, [
                sys.executable, str(HERE.parent / "llama-table1-825/llama_table1.py"),
                "--serve-prm", "--prm-port", str(PORTS[lane + 2]),
            ], env, args.output / f"{label}-prm.log")
            llm = start_service(stack, [
                args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve",
                str(released.runner.asset_path(model)), "--host", "127.0.0.1", "--port", str(PORTS[lane]),
                "--served-model-name", model[1], *ENGINE_FLAGS,
            ], {k: v for k, v in env.items() if k != "PYTHONPATH"}, args.output / f"{label}-engine.log")
            services.extend(((prm, PORTS[lane + 2]), (llm, PORTS[lane])))
        for process, port in services:
            wait_ready(process, port)
        for lane, label in enumerate(MODELS):
            worker_process = start_service(stack, [
                sys.executable, str(Path(__file__)), "--model", label, "--vllm-version", version,
                "--data", str(args.data), "--output", str(args.output / label),
            ], {**os.environ, "CUDA_VISIBLE_DEVICES": devices[lane]}, args.output / f"{label}-worker.log")
            workers.append((label, worker_process))
        codes = {label: worker_process.wait() for label, worker_process in workers}
        if any(codes.values()):
            raise RuntimeError(f"Candidate generation failed: {codes}")
    released.write(args.output / "generation.json", {"wall_seconds": time.monotonic() - started,
                                                      "worker_exit_codes": codes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-version")
    parser.add_argument("--engine-python")
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data = args.data or args.output.parent / "data"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    worker(args) if args.model else coordinate(args)


if __name__ == "__main__":
    main()

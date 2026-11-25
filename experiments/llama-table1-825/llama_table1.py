"""Run and audit the published-code Llama experiments on two allocated H20s."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import build_opener, ProxyHandler

import run_published_extensions as released
from audit_results import check_problem


MODEL = ("model", "meta-llama/Llama-3.1-8B-Instruct", "359efdbb8af05b788a4ad4185215c6b8caa9052c")
DATASETS = ("math", "aime24", "amo")
PAPER_COUNTS = {"math": (219, 223, 236, 327), "aime24": (1, 2, 2, 3), "amo": (1, 1, 2, 2)}
WORKERS = {"math": (64, 48, 64, 24), "aime24": (16, 16, 16, 12), "amo": (16, 16, 16, 12)}
CONTEXT = 32768
opener = build_opener(ProxyHandler({}))
released.runner.ASSETS = (released.runner.ASSETS[0], MODEL, released.runner.ASSETS[2])


def wait_ready(process, port):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Service on port {port} exited with {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Service on port {port} did not become ready")


def start_service(stack, command, env, log):
    output = stack.enter_context(log.open("a", encoding="utf-8"))
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)

    def stop():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    stack.callback(stop)
    return process


def run_shard(args, dataset, case, lane, workers, devices, ports, version, sample):
    out = args.output / ("smoke" if sample else "formal") / dataset / f"gpu-{lane}" / case
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__)), "--worker", case,
        "--dataset", dataset, "--manifest", str(args.manifest), "--output", str(out),
        "--lane", str(lane), "--gen-port", str(ports[lane]), "--prm-port", str(ports[lane + 2]),
        "--workers", str(workers), "--vllm-version", version,
    ]
    if sample is not None:
        cmd.extend(("--sample", str(sample)))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[lane]}
    with (out / "worker.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return out, result.returncode


def audit(args, dataset, case, questions, paths, version, paper):
    rows, configs, shards = [], [], []
    iterations = released.CASES[case][1]
    for lane, path in enumerate(paths):
        config = released.read(path / "config.json")
        summary = released.read(path / "summary.json")
        expected = list(range(lane, len(questions), 2))
        predictions = released.read(path / "predictions.json")
        assert len(predictions) == len(expected) and [r["problem_id"] for r in predictions] == expected
        assert config["sample_ids"] == expected and config["dataset"] == dataset and config["seed"] == 0
        assert config["upstream_commit"] == released.runner.UPSTREAM_COMMIT
        assert config["eval"]["num_iterations"] == iterations
        assert list(MODEL) in config["datasets_and_models"]
        assert summary["samples"] == len(expected) and not summary["failed"]
        for prediction in predictions:
            index = prediction["problem_id"]
            problem = path / "problems" / f"{index:03d}"
            record = released.read(problem / "result.json")
            check_problem(problem, record, index, iterations, released.CLASSIFICATION)
            assert record["dataset_unique_id"] == prediction["dataset_unique_id"] == questions[index]["dataset_unique_id"]
            assert record["problem"] == questions[index]["problem"]
            assert record["gold_answer"] == questions[index]["gold_answer_extracted"]
            assert record["final_is_correct"] == prediction["is_correct"]
            assert record["final_predicted_answer"] == prediction["predicted_answer"]
            rows.append({**prediction, "lane": lane,
                         "candidate_oracle_correct": any(c["is_correct"] is True for c in record["iterations"])})
        assert summary["correct"] == sum(p["is_correct"] is True for p in predictions)
        summary.pop("failed", None)
        shards.append(summary)
        config["runtime"]["cuda_visible_devices"] = str(lane)
        config["client"]["model_id"] = "/".join(MODEL[1:])
        config["dataset_manifest"] = "datasets/manifest.json"
        configs.append(config)
    rows.sort(key=lambda row: row["problem_id"])
    assert [row["problem_id"] for row in rows] == list(range(len(questions)))
    target = args.output / "export" / dataset / case
    target.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    (target / "predictions.jsonl").write_bytes(payload)
    correct = sum(row["is_correct"] is True for row in rows)
    result = {"dataset": dataset, "case": case, "samples": len(rows), "correct": correct,
              "accuracy": correct / len(rows), "paper_correct": paper, "vllm": version,
              "upstream_commit": released.runner.UPSTREAM_COMMIT,
              "predictions_sha256": hashlib.sha256(payload).hexdigest(),
              "configs": configs, "shards": shards}
    released.write(target / "audit.json", result)
    return {key: result[key] for key in ("samples", "correct", "accuracy", "paper_correct", "predictions_sha256")}


def coordinate(args):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or not all(devices):
        raise RuntimeError("Expected exactly the two Slurm-assigned devices in CUDA_VISIBLE_DEVICES")
    if int(os.environ.get("SLURM_GPUS_ON_NODE", "2")) != 2:
        raise RuntimeError("Expected a two-GPU allocation")
    expected = released.runner.asset_path(MODEL)
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))"
    ], text=True).strip()
    if version != "0.33.0":
        raise RuntimeError(f"Expected vLLM 0.33.0, found {version}")
    first_port = 22500 + 4 * (int(os.environ.get("SLURM_JOB_ID", "0")) % 7000)
    ports = tuple(range(first_port, first_port + 4))
    for port in ports:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "status.json"
    released.write(status, {"phase": "starting_services"})
    started = time.perf_counter()
    results = {}
    pipeline_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with ExitStack() as stack:
        processes = []
        for lane, device in enumerate(devices):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            prm = start_service(stack, [sys.executable, str(Path(__file__)), "--serve-prm",
                                        "--prm-port", str(ports[lane + 2])],
                                env, args.output / f"prm-{lane}.log")
            engine = start_service(stack, [
                args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve", str(expected),
                "--host", "127.0.0.1", "--port", str(ports[lane]), "--served-model-name", MODEL[1],
                "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--generation-config", "vllm",
                "--gpu-memory-utilization", "0.60", "--max-model-len", str(CONTEXT),
                "--enable-prefix-caching", "--max-num-seqs", "112", "--max-num-batched-tokens", "8192",
            ], {key: value for key, value in env.items() if key != "PYTHONPATH"}, args.output / f"engine-{lane}.log")
            processes.extend(((prm, ports[lane + 2]), (engine, ports[lane])))
        for process, port in processes:
            wait_ready(process, port)
        with ThreadPoolExecutor(max_workers=2) as pool:
            def run_pair(dataset, case, workers, sample=None):
                released.write(status, {"phase": "smoke" if sample else "formal",
                                        "dataset": dataset, "case": case, "audited": results})
                pending = pool.map(lambda lane: run_shard(args, dataset, case, lane, workers, devices, ports,
                                                          version, sample), (0, 1))
                paths = []
                for lane, (path, code) in enumerate(pending):
                    if code:
                        path, code = run_shard(args, dataset, case, lane, workers, devices, ports, version, sample)
                    if code:
                        raise RuntimeError(f"Incomplete {dataset} {case} lane {lane}, see {path / 'worker.log'}")
                    summary = released.read(path / "summary.json")
                    if not summary["samples"] or summary["failed"]:
                        raise RuntimeError(f"Empty or failed {dataset} {case} lane {lane}")
                    paths.append(path)
                return paths

            run_pair("math", "E1", 4, sample=8)
            for dataset in DATASETS:
                run_pair(dataset, "E4", 4, sample=8)
            for dataset in DATASETS:
                questions, _ = released.load_benchmark(SimpleNamespace(dataset=dataset, dataset_manifest=args.manifest))
                results[dataset] = {}
                for pos, case in enumerate(released.CASES):
                    workers = min(WORKERS[dataset][pos], len(questions) // 2)
                    paths = run_pair(dataset, case, workers)
                    results[dataset][case] = audit(args, dataset, case, questions, paths, version, PAPER_COUNTS[dataset][pos])
                    print(f"{dataset} {case} verified {results[dataset][case]['correct']}/{len(questions)}", flush=True)
    released.write(args.output / "export" / "summary.json", {
        "model": MODEL, "seconds": time.perf_counter() - started, "cases": results,
        "pipeline_sha256": pipeline_hash, "total_predictions": sum(
            ablation["samples"] for dataset in results.values() for ablation in dataset.values()),
    })
    released.write(status, {"phase": "verified", "audited": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", choices=released.CASES)
    mode.add_argument("--serve-prm", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--engine-python")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--lane", type=int, choices=(0, 1))
    parser.add_argument("--gen-port", type=int)
    parser.add_argument("--prm-port", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--vllm-version")
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()
    if args.serve_prm:
        import serve_prm
        serve_prm.ADDRESS = ("127.0.0.1", args.prm_port)
        serve_prm.serve()
        return 0
    if args.worker:
        from answer_fields import install

        install()
        worker_args = SimpleNamespace(
            case=args.worker, backend="vllm", dataset=args.dataset, dataset_manifest=args.manifest,
            prm_port=args.prm_port, workers=args.workers, sample=args.sample, seed=0, port=args.gen_port,
            context=CONTEXT, model=MODEL[1], vllm_version=args.vllm_version, output=args.output,
            shards=2, shard=args.lane, priority_file=None, strict_errors=False,
        )
        import fcntl

        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".worker.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return 0 if released.run(worker_args) else 1
    if not args.output or not args.manifest or not args.engine_python:
        parser.error("Coordinator needs --output, --manifest and --engine-python")
    try:
        coordinate(args)
    except BaseException as error:
        if args.output.is_dir():
            released.write(args.output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    return 0


if __name__ == "__main__":
    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())

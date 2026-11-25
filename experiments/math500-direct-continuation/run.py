"""Run paired Direct token-limit replay, then a small-pool voting baseline."""
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

from protocol import DRAWS, LONG, SHORT, STUDIES


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "gsm8k-500-e4-direct"
sys.path.insert(0, str(PRIOR))
from evaluate import (
    CONTEXT, DIRECT_KEYS, DIRECT_WORKERS, ENGINE, ENGINE_FLAGS, LANES, MODELS, MeasuredSession,
    assigned, released, sha, start_service, wait_ready,
)
from sample import load as load_gsm8k, plan as gsm8k_plan


def load(data, study):
    runner = released.runner
    cmd = ["git", "-C", str(runner.ROOT / "upstream")]
    if (subprocess.check_output([*cmd, "rev-parse", "HEAD"], text=True).strip() != runner.UPSTREAM_COMMIT
            or subprocess.check_output([*cmd, "status", "--porcelain", "--untracked-files=no"])):
        raise ValueError("Expected the clean pinned evaluator and Direct prompts")
    rows, spec = released.load_benchmark(SimpleNamespace(dataset="math"))
    freeze = released.read(PRIOR / "followup_questions.json")
    if spec["sha256"] != freeze["sources"]["math"]["sha256"] or len(rows) != 500:
        raise ValueError("Unexpected MATH-500 source")
    questions = []
    for i, q in enumerate(rows):
        record = freeze["questions"][i]
        if ((record["problem_id"], record["dataset_unique_id"], record["gold_answer_extracted"],
             record["question_sha256"]) != (i, q["dataset_unique_id"], q["gold_answer_extracted"],
                                             sha(q["problem"].encode("utf-8")))):
            raise ValueError(f"Changed MATH-500 source identity: {i}")
        questions.append({**q, "problem_id": i, "dataset": "math", "seed": i})
    if study == "voting":
        gs = load_gsm8k(data)
        if gsm8k_plan(gs) != released.read(PRIOR / "questions.json"):
            raise ValueError("Unexpected GSM8K sample")
        questions.extend({**q, "seed": q["problem_id"], "problem_id": len(rows) + q["problem_id"]} for q in gs)
    return questions


class CapturingSession(MeasuredSession):
    def __init__(self):
        super().__init__()
        self.choice = None

    def post(self, url, *args, **kwargs):
        if not url.endswith("/v1/completions"):
            raise ValueError("Direct generation only accepts completion requests")
        response = super().post(url, *args, **kwargs)
        if response.ok:
            if self.choice is not None:
                raise ValueError("Expected one successful generation")
            self.choice, = response.json()["choices"]
        return response


def one(q, model_args, cfg, tokenizer, decode, evaluator, prompts, draw):
    messages = [
        {"role": "system", "content": prompts[DIRECT_KEYS[0]]},
        {"role": "user", "content": prompts[DIRECT_KEYS[1]].format(problem=q["problem"])},
    ]
    seed = q["seed"] + draw * 1_000_000
    with CapturingSession() as session:
        session.stage("direct")
        client = released.TransformerWrapper(released.runner.upstream.DictConfig(cfg["client"]))
        client._initialized = True
        client.tokenizer = tokenizer
        client.model = released.GenerationProxy(model_args, decode, seed, [], session)
        completion = client.generate(messages).completion
        ids = session.choice["token_ids"]
        finish = session.choice["finish_reason"]
        costs = session.stages[0]
    if (not ids or len(ids) != costs["output_tokens"] or not costs["input_tokens"]
            or len(ids) > decode["max_tokens"] or finish not in ("stop", "length")
            or (finish == "length" and len(ids) != decode["max_tokens"])
            or costs["generation_requests"] - costs["generation_without_usage"] != 1 or costs["prm_requests"]):
        raise ValueError(f"Invalid Direct ledger: {q['problem_id']}/{draw}")

    def grade(text, n):
        answer = evaluator._extract_answer_from_model_completion(text)
        return {"answer": answer, "correct": evaluator._check_answers_equiv(answer, q["gold_answer_extracted"]) is True,
                "output_tokens": n, "completion_sha256": sha(text.encode("utf-8"))}

    full = grade(completion, len(ids))
    record = {"seed": seed, **costs, "finish_reason": finish}
    if decode["max_tokens"] == LONG:
        prefix = tokenizer.decode(ids[:SHORT], skip_special_tokens=True) if len(ids) > SHORT else completion
        record.update(token_ids=ids, full=full, short=grade(prefix, min(SHORT, len(ids))))
    else:
        record.update(completion=completion, **full)
    return record


def worker(args):
    label, port, _ = LANES[args.lane]
    model = MODELS[label]
    runner = released.runner
    assets, _, prm = runner.ASSETS
    runner.ASSETS = (assets, model, prm)
    cfg = released.configuration("E4")
    default = cfg["client"]["generate_kwargs"]["max_tokens"]
    if default != SHORT:
        raise ValueError("Expected the prior 1024-token Direct cap")
    cfg["client"]["generate_kwargs"]["max_tokens"] = LONG if args.study == "continuation" else SHORT
    raw = released.AutoTokenizer.from_pretrained(cfg["client"]["model_id"], local_files_only=True)
    tokenizer = released.LockedTokenizer(raw)
    decode = released.effective_decode(cfg, raw)
    prompts = runner.upstream.get_prompts()
    evaluator = runner.upstream.get_evaluator("MATH")
    model_args = SimpleNamespace(port=port, model=model[1], context=CONTEXT)
    questions = assigned(load(args.data, args.study), args.lane)
    path = args.output / "problems"
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "lane": args.lane, "model": list(model), "decode": decode, "study": args.study,
        "upstream_commit": runner.UPSTREAM_COMMIT, "concurrency": DIRECT_WORKERS,
        "engine": {**ENGINE, "enable-prefix-caching": True, "vllm": "0.33.0"},
        "prompts": {key: prompts[key] for key in DIRECT_KEYS}, "source_sha256": sha(Path(__file__).read_bytes()),
        "ids": [q["problem_id"] for q in questions],
    }
    cfg_path = args.output / "config.json"
    if cfg_path.exists() and released.read(cfg_path) != config:
        raise ValueError("Cannot resume with different generation settings")
    released.write(cfg_path, config)

    def evaluate(q):
        identity = {k: q[k] for k in ("problem_id", "dataset", "dataset_unique_id")}
        n = int(q["problem_id"])
        target = path / f"{n:04d}.json"
        required = 1 if args.study == "continuation" else DRAWS
        if target.exists():
            row = released.read(target)
            if any(row[k] != v for k, v in identity.items()) or len(row["outputs"]) != required:
                raise ValueError("Checkpoint identity differs")
            return
        completions = [
            one(q, model_args, cfg, tokenizer, decode, evaluator, prompts, draw) for draw in range(required)
        ]
        released.write(target, {**identity, "lane": args.lane, "gold": q["gold_answer_extracted"],
                                "outputs": completions})

    with (args.output / "upstream.log").open("a", encoding="utf-8", buffering=262144) as log:
        with redirect_stdout(log), redirect_stderr(log), ThreadPoolExecutor(max_workers=DIRECT_WORKERS) as pool:
            jobs = [pool.submit(evaluate, q) for q in questions]
            for count, job in enumerate(as_completed(jobs), 1):
                job.result()
                if count % 50 == 0 or count == len(jobs):
                    print(f"{args.study} · GPU {args.lane} · {count}/{len(jobs)}", file=sys.__stdout__, flush=True)


def coordinate(args):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(LANES) or len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three distinct GPUs")
    version = subprocess.check_output([
        args.engine_python, "-c", "from importlib.metadata import version; print(version('vllm'))",
    ], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Unexpected vLLM version {version}")
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as services:
        servers = []
        for lane, device in zip(LANES, devices):
            if lane == 7:
                continue
            model = MODELS[LANES[lane][0]]
            port = LANES[lane][1]
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["CUDA_VISIBLE_DEVICES"] = device
            child = start_service(services, [
                args.engine_python, "-m", "vllm.entrypoints.cli.main", "serve",
                str(released.runner.asset_path(model)), "--host", "127.0.0.1", "--port", str(port),
                "--served-model-name", model[1], *ENGINE_FLAGS,
            ], env, args.output / f"gpu-{lane}-engine.log")
            servers.append((child, port))
        for child, port in servers:
            wait_ready(child, port)
        for study in STUDIES:
            target = args.output / study
            target.mkdir(exist_ok=True)
            released.write(args.output / "status.json", {"phase": study})
            started = time.monotonic()
            with ExitStack() as jobs:
                processes = {
                    str(lane): start_service(jobs, [
                        sys.executable, "-B", str(Path(__file__)), "--lane", str(lane), "--study", study,
                        "--data", str(args.data), "--output", str(target / f"gpu-{lane}"),
                    ], {**os.environ, "CUDA_VISIBLE_DEVICES": device}, target / f"gpu-{lane}-worker.log")
                    for lane, device in zip(LANES, devices)
                }
                codes = {lane: proc.wait() for lane, proc in processes.items()}
                if any(codes.values()):
                    raise RuntimeError(f"{study} failed: {codes}")
            released.write(target / "generation.json", {"worker_exit_codes": codes, "wall_seconds": time.monotonic() - started})
    released.write(args.output / "status.json", {"phase": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-python")
    parser.add_argument("--lane", type=int, choices=tuple(LANES))
    parser.add_argument("--study", choices=STUDIES)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    worker(args) if args.lane is not None else coordinate(args)

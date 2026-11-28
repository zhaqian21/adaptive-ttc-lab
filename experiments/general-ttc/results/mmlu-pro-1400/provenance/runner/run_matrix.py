"""Run the three MMLU-Pro model/scorer pairs on the single AutoDL GPU."""
from datetime import datetime
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

queue = Path(__file__).resolve().parent
root = queue.parent
next_models = root / "models-next"
source = root / "concurrent-source"
runner = Path("/root/ttc-vgpu48-runtime/e1-prm/bin/python")
engine = Path("/root/ttc-vgpu48-runtime/e1-engine/bin")
cache = root.parent / "vgpu48/cache"
manifest = root / "mmlu-pro-1400.json"
manifest_sha = "9246bdbb6f285f89babb74cbefa6cd3f8e443f79bf798130c867ab2d4b509dd0"
revision = "2046e098ed83869a4d71fe772cc18b62336ac3de"
generator_url = "http://127.0.0.1:18420/v1"
scorer_url = "http://127.0.0.1:18423"
models = {
    "llama": ("published-llama31", next_models / "Llama-3.1-8B-Instruct/359efdbb8af05b788a4ad4185215c6b8caa9052c"),
    "qwen": ("published-qwen25", root.parent / "vgpu48/models/Qwen2.5-7B-Instruct"),
}
pairs = [("llama", "versa"), ("qwen", "dprm"), ("llama", "dprm")]
methods = {"E1": 96, "E2": 96, "E3": 112, "E4": 96}
opener = build_opener(ProxyHandler({}))


def log(text):
    print(datetime.now().astimezone().isoformat(timespec="seconds"), text, flush=True)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)


def stage(text):
    write(queue / "stage", text + "\n")
    log(text)


def json_request(url, body=None, timeout=4):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def get(url):
    try:
        return json_request(url)
    except (OSError, URLError, ValueError):
        return {}


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    except FileNotFoundError:
        return []


def pid_of(path):
    return int(path.read_text().strip()) if path.is_file() else None


def wait_for(description, check, seconds=420, pid=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        if pid and not any(cmdline(pid)):
            raise RuntimeError(f"{description}: process {pid} exited, see service log")
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for {description}")


def stop(pidfile, port, checks, group=False):
    pid = pid_of(pidfile)
    args = cmdline(pid)
    if any(args):
        if not checks(args):
            raise RuntimeError(f"Refusing to stop unexpected process {pid}")
        if group and os.getpgid(pid) != pid:
            raise RuntimeError(f"Generator {pid} is outside its dedicated process group")
        log(f"Stopping project service {pid} on port {port}")
        if group:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        wait_for(f"service {pid} to terminate", lambda: not any(cmdline(pid)), seconds=80)
    wait_for(
        f"port {port} to release",
        lambda: not get(generator_url + "/models" if port == 18420 else scorer_url + "/health"),
        seconds=45,
    )


def launch(args, logfile, pidfile, env):
    with logfile.open("ab") as handle:
        p = subprocess.Popen(
            args, cwd=root, env={**os.environ, **env}, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    write(pidfile, f"{p.pid}\n")
    return p.pid


def generator(model):
    label, model_path = models[model]
    pidfile = queue / "generator.pid"

    def ready():
        data = get(generator_url + "/models").get("data", [])
        return any(entry["id"] == label for entry in data)

    if ready():
        return label
    pid = pid_of(pidfile)
    if label in cmdline(pid):
        wait_for(label, ready, pid=pid)
        return label
    stop(pidfile, 18420, lambda args: "serve" in args and "18420" in args and any(
        str(path) in args for _, path in models.values()), group=True)
    cmd = [
        str(engine / "vllm"), "serve", str(model_path),
        "--host", "127.0.0.1", "--port", "18420", "--served-model-name", label,
        "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "32768",
        "--max-num-seqs", "112", "--max-num-batched-tokens", "8192",
        "--gpu-memory-utilization", "0.60", "--enable-prefix-caching", "--generation-config", "vllm",
    ]
    env = {
        "PATH": str(engine) + ":" + os.environ["PATH"],
        "CUDA_HOME": "/usr/local/cuda-13.0", "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "HF_HOME": str(cache / "hf"), "XDG_CACHE_HOME": str(cache / "xdg"),
        "VLLM_CACHE_ROOT": str(cache / "e1-vllm"), "TRITON_CACHE_DIR": str(cache / "e1-triton"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "e1-flashinfer"),
    }
    pid = launch(cmd, queue / f"{model}-vllm.log", pidfile, env)
    wait_for(label, ready, pid=pid)
    return label


def ready_scorer(name):
    health = get(scorer_url + "/health")
    assets = health.get("assets", {})
    if health.get("status") != "ready":
        return {}
    if name == "versa" and {"prm_base", "prm_adapter"} <= assets.keys():
        return health
    if name == "dprm" and assets.get("adapter", {}).get("repo") == "dongboklee/dPRM-qwen":
        return health
    return {}


def scorer(name):
    if ready_scorer(name):
        return
    if name == "dprm":
        stage("Checking verified dPRM weights and isolated dependencies")
        for status in (next_models / "downloads/dprm.exit", queue / "dprm-deps.exit"):
            wait_for(status.name, status.is_file, seconds=900)
            if status.read_text().strip() != "0":
                raise RuntimeError(f"Prerequisite failed: {status}")
    pidfile = queue / "scorer.pid"
    if not pidfile.is_file():
        pidfile = root / "logs/versaprm.pid"
    stop(pidfile, 18423, lambda args: any(
        Path(arg).name in {"serve_prm.py", "serve_dprm.py"} for arg in args
    ), group=False)
    env = {
        "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(root / "python-deps") + ":" + str(source),
    }
    if name == "dprm":
        env["PYTHONPATH"] = str(queue / "dprm-deps") + ":" + env["PYTHONPATH"]
        cmd = [str(runner), "-u", str(queue / "serve_dprm.py"), "--models", str(next_models), "--verify"]
    else:
        cmd = [str(runner), "-u", str(source / "serve_prm.py"), "--models", str(root / "models")]
    pid = launch(cmd, queue / f"{name}-prm.log", queue / "scorer.pid", env)
    wait_for(name, lambda: ready_scorer(name), pid=pid)
    check = json_request(scorer_url + "/score", {
        "question": "Question: What is 2 + 3?\nOptions:\n(A) 5\n(B) 6",
        "steps": ["Add 2 and 3 to get 5.", "The answer is (A) 5."],
    }, timeout=120)
    assert len(check["step_scores"]) == 2 and all(
        math.isfinite(score) and 0 <= score <= 1 for score in check["step_scores"])
    log(f"{name} scorer passed /score smoke")


def validate(output, method, model, expected, rm):
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert config["generator"] == model and config["method"] == method
    assert config["manifest_sha256"] == manifest_sha and config["questions"] == list(expected)
    assert (config["seed"], config["max_tokens"], config["search_steps"], config["fixed_strategy"]) == (0, 1024, 2, None)
    assert config["outer_k"] == (10 if method in {"E2", "E4"} else 1)
    assert config["scorer"] == (None if method == "E1" else ready_scorer(rm))
    counts = {}
    records = {}
    for file in (output / "questions").glob("*.json"):
        row = json.loads(file.read_text())
        task = row["task_id"]
        assert task not in records
        assert (row["category"], row["gold"]) == expected[task]
        records[task] = row
        category = row["category"]
        counts[category] = counts.get(category, 0) + 1
    assert set(records) == set(expected) and len(records) == summary["questions"]
    assert len(counts) == 14 and set(counts.values()) == {100}
    for key in ("correct", "prm_calls", "generator_calls", "input_tokens", "output_tokens", "prm_input_tokens"):
        assert summary[key] == sum(row[key] for row in records.values())
    if method in {"E2", "E4"}:
        assert summary["prm_calls"] > 0
        assert all(len(row["candidates"]) == 10 for row in records.values())
    return summary


def eval_run(pair, method, model, output, workers, log_file, limit=None):
    flags = []
    if (output / "config.json").is_file():
        flags.append("--resume")
    if limit:
        flags.extend(["--limit", str(limit)])
    command = [
        str(runner), "-u", str(source / "run.py"), "--manifest", str(manifest), "--method", method,
        "--generator-url", generator_url, "--generator-model", model, "--prm-url", scorer_url,
        "--seed", "0", "--k", "10", "--max-tokens", "1024", "--search-steps", "2",
        "--workers", str(workers), "--code-revision", revision, "--output", str(output), *flags,
    ]
    with log_file.open("ab") as handle:
        log(f"{pair} {method} {'smoke' if limit else 'full'} c{workers}")
        return subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT).returncode


def aggregate():
    rows = []
    for folder in (root / "full", queue / "results"):
        if folder == root / "full":
            summaries = {f"{method}-1400/summary.json": ("qwen-versa", method) for method in methods}
        else:
            summaries = {
                f"{g}-{s}/{m}/summary.json": (f"{g}-{s}", m)
                for g, s in pairs for m in methods
            }
        for relative, (pair, method) in summaries.items():
            path = folder / relative
            if path.is_file():
                report = json.loads(path.read_text())
                rows.append({"pair": pair, "method": method, **report})
    write(queue / "matrix.json", json.dumps(rows, indent=2) + "\n")
    fields = ["pair", "method", "questions", "correct", "accuracy", "generator_calls", "input_tokens",
              "output_tokens", "prm_calls", "prm_input_tokens", "elapsed_seconds"]
    temp = queue / "matrix.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as handle:
        report = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        report.writeheader()
        report.writerows(rows)
    temp.replace(queue / "matrix.csv")


def main():
    encoded = manifest.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == manifest_sha
    expected = {r["id"]: (r["category"], r["answer"]) for r in json.loads(encoded)["questions"]}
    assert len(expected) == 1400
    assert (root / "queue-e3-e2-e4/exit").read_text().strip() == "0"
    for gen, rm in pairs:
        pair = f"{gen}-{rm}"
        stage(f"Preparing {pair}")
        model = generator(gen)
        scorer(rm)
        report = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"
        ], text=True)
        write(queue / f"{pair}-services.json", json.dumps({
            "generator": get(generator_url + "/models"), "scorer": ready_scorer(rm), "gpu": report.strip(),
        }, indent=2) + "\n")
        for method, workers in methods.items():
            if pair == "llama-versa" and method == "E4":
                workers = 112
            output = queue / "results" / pair / method
            if not (output / "summary.json").is_file():
                if method in {"E1", "E2"}:
                    smoke = queue / "smoke" / pair / method
                    if not (smoke / "summary.json").is_file():
                        stage(f"Smoke {pair} {method}")
                        rc = eval_run(pair, method, model, smoke, 3, queue / f"{pair}-{method}-smoke.log", limit=3)
                        if rc:
                            raise RuntimeError(f"{pair} {method} smoke failed")
                for concurrency in (workers, 48, 24):
                    stage(f"Running {pair} {method} c{concurrency}")
                    rc = eval_run(pair, method, model, output, concurrency, queue / f"{pair}-{method}.log")
                    if rc == 0:
                        break
                    if not (output / "config.json").is_file():
                        raise RuntimeError(f"{pair} {method} failed before checkpoints were prepared")
                    log(f"{pair} {method} will resume completed checkpoints")
                    time.sleep(8)
                else:
                    raise RuntimeError(f"{pair} {method} failed after retries")
            values = validate(output, method, model, expected, rm)
            log(f"{pair} {method}: {values['correct']}/{values['questions']}, PRM {values['prm_calls']}")
            aggregate()
    stage("Complete: all three model/scorer pairs E1–E4 verified")


if __name__ == "__main__":
    with (queue / "queue.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as error:
            stage(f"Paused: {error}")
            raise

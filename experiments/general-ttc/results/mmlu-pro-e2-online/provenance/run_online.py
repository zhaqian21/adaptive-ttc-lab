"""Run the four online E2 conditions on the existing AutoDL Qwen/dPRM services."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

runtime = Path(__file__).resolve().parent
root = runtime.parent
repo = root / "e2-online-publication"
manifest = root / "mmlu-pro-1400.json"
script = repo / "experiments/general-ttc/run.py"
arms = (
    ("k10-mean", "mean", None),
    ("k10-min", "min", None),
    ("stop5-mean", "mean", 5),
    ("stop5-min", "min", 5),
)
frozen_sha = "9246bdbb6f285f89babb74cbefa6cd3f8e443f79bf798130c867ab2d4b509dd0"
spec = importlib.util.spec_from_file_location("service_setup", root / "extension-mmlu-pro/run_matrix.py")
services = importlib.util.module_from_spec(spec)
spec.loader.exec_module(services)
services.queue = runtime


def execute(output, log_path, revision, aggregation, agreement, workers, limit=None):
    command = [
        sys.executable, "-B", "-u", str(script), "--manifest", str(manifest), "--method", "E2",
        "--generator-model", "published-qwen25", "--generator-url", services.generator_url,
        "--prm-url", services.scorer_url, "--seed", "0", "--k", "10", "--max-tokens", "1024",
        "--search-steps", "2", "--workers", str(workers), "--code-revision", revision,
        "--prm-aggregation", aggregation, "--output", str(output),
    ]
    if agreement is not None:
        command.extend(("--agreement-stop", str(agreement)))
    if limit is not None:
        command.extend(("--limit", str(limit)))
    if (output / "config.json").is_file():
        command.append("--resume")
    with log_path.open("ab") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode


def collect_service(health, revision):
    env = {
        "dataset_sha256": frozen_sha,
        "code_revision": revision,
        "generator": "published-qwen25",
        "generator_flags": services.cmdline(services.pid_of(runtime / "generator.pid"))[2:],
        "scorer": health,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "workers": 96, "warmup_workers": 8, "warmup_questions": 8, "warmup_policy": "k10-mean",
    }
    services.write(runtime / "environment.json", json.dumps(env, indent=2) + "\n")


def verify_arm(output, aggregate, agreement):
    summary = json.loads((output / "summary.json").read_text())
    assert summary["questions"] == 1400 and summary["manifest_sha256"] == frozen_sha
    assert summary["prm_aggregation"] == aggregate and summary["agreement_stop"] == agreement
    assert summary["generator_calls"] == 14000 - summary["early_stopped"] * 5
    assert summary["prm_calls"] <= 14000 - summary["early_stopped"] * 10
    return summary


def main():
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == frozen_sha
    assert (root / "extension-mmlu-pro/exit").read_text().strip() == "0"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    health = services.ready_scorer("dprm")
    if not health:
        raise RuntimeError("The original dPRM service is unavailable")
    for arm, rule, agreement in arms:
        output = runtime / "full" / arm
        if (output / "summary.json").is_file():
            verify_arm(output, rule, agreement)
            continue
        services.stage(f"Preparing {arm}")
        services.stop(runtime / "generator.pid", 18420, lambda cmd: (
            "serve" in cmd and "18420" in cmd and str(services.models["qwen"][1]) in cmd
        ), group=True)
        services.generator("qwen")
        if not (runtime / "environment.json").is_file():
            collect_service(health, revision)
        warmup = runtime / "warmup" / arm
        if (warmup / "summary.json").is_file():
            warmup = warmup.with_name(f"{warmup.name}-{int(time.time())}")
        services.stage(f"Warmup {arm}")
        status = execute(warmup, runtime / f"{arm}-warmup.log", revision, "mean", None, 8, 8)
        if status:
            raise RuntimeError(f"Warmup failed for {arm}")
        for attempt in range(3):
            services.stage(f"Running {arm} c96")
            status = execute(output, runtime / f"{arm}.log", revision, rule, agreement, 96)
            if not status:
                break
            if not (output / "config.json").is_file() or not services.ready_scorer("dprm"):
                raise RuntimeError(f"Unable to resume {arm}")
            services.log(f"Retrying unfinished questions for {arm} ({attempt+1}/3)")
            time.sleep(8)
        else:
            raise RuntimeError(f"{arm} still has unfinished questions")
        summary = verify_arm(output, rule, agreement)
        services.log(f"{arm} {summary['correct']}/1400, {summary['early_stopped']} early, {summary['elapsed_seconds']:.1f}s")
    services.stage("Complete: four online E2 conditions")


if __name__ == "__main__":
    with (runtime / "queue.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except Exception as error:
            services.stage(f"Paused: {error}")
            raise

"""Run three additional E2 model and scorer pairs on the single AutoDL GPU."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from archive_transfer import ARMS, FROZEN_SHA, PAIRS, archive_pair, write_comparison


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def execute(output, log_path, revision, aggregation, agreement, workers, limit=None, *, generator_model, framework, manifest, services):
    command = [
        sys.executable, "-B", "-u", str(framework / "run.py"), "--manifest", str(manifest), "--method", "E2",
        "--generator-model", generator_model, "--generator-url", services.generator_url,
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


def run_matrix(runtime):
    project = runtime.parent
    framework = Path(__file__).resolve().parent.parent
    services = module(project / "extension-mmlu-pro/run_matrix.py", "autodl_services")
    services.queue = runtime
    services.source = framework
    manifest = project / "mmlu-pro-1400.json"
    data = manifest.read_bytes()
    if hashlib.sha256(data).hexdigest() != FROZEN_SHA:
        raise RuntimeError("Question set differs from PR #46")
    for prior in ("extension-mmlu-pro", "e2-online-ablation"):
        if (project / prior / "exit").read_text().strip() != "0":
            raise RuntimeError(f"{prior} has an unfinished experiment")
    revision = (runtime / "code-revision").read_text().strip()
    expected = {q["id"]: (q["category"], q["answer"]) for q in json.loads(data)["questions"]}

    with (runtime / "queue.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for combo, gen, prm in PAIRS:
                health_path = framework / "results/mmlu-pro-1400/runs" / combo / "E2/config.json"
                expected_scorer = json.loads(health_path.read_text())["scorer"]
                alias, _ = services.models[gen]
                for arm, rule, stop in ARMS:
                    output = runtime / "full" / combo / arm
                    if (output / "summary.json").is_file():
                        continue
                    logdir = runtime / "records" / combo
                    logdir.mkdir(parents=True, exist_ok=True)
                    services.stage(f"Preparing {combo} {arm}")
                    services.stop(runtime / "generator.pid", 18420, lambda cmd: (
                        "serve" in cmd and "18420" in cmd and any(
                            str(path) in cmd for _, path in services.models.values()
                        )
                    ), group=True)
                    services.scorer(prm)
                    services.generator(gen)
                    health = services.ready_scorer(prm)
                    if health != expected_scorer:
                        raise RuntimeError(f"Unexpected scorer metadata for {combo}")
                    config = {
                        "generator": alias, "scorer": health, "code_revision": revision,
                        "generator_flags": services.cmdline(services.pid_of(runtime / "generator.pid"))[2:],
                        "gpu": subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                            text=True).strip(),
                        "manifest_sha256": FROZEN_SHA, "workers": 96,
                        "warmup_workers": 8, "warmup_questions": 8, "warmup_policy": "k10-mean",
                    }
                    services.write(logdir / f"{arm}-services.json", json.dumps(config, indent=2) + "\n")
                    warmup = runtime / "warmup" / combo / arm
                    if (warmup / "summary.json").is_file():
                        warmup = warmup.with_name(f"{arm}-{int(time.time())}")
                    services.stage(f"Warmup {combo} {arm}")
                    rc = execute(warmup, logdir / f"{arm}-warmup.log", revision, "mean", None,
                                  8, 8, generator_model=alias, framework=framework,
                                  manifest=manifest, services=services)
                    if rc:
                        raise RuntimeError(f"Warmup failed for {combo} {arm}")
                    services.write(logdir / f"{arm}-warmup.json", (warmup / "summary.json").read_text())
                    for attempt in range(3):
                        services.stage(f"Running {combo} {arm} c96")
                        rc = execute(output, logdir / f"{arm}.log", revision, rule, stop,
                                      96, generator_model=alias, framework=framework,
                                  manifest=manifest, services=services)
                        if not rc:
                            break
                        if not (output / "config.json").is_file() or not services.ready_scorer(prm):
                            raise RuntimeError(f"Unable to resume {combo} {arm}")
                        services.log(f"Retrying unfinished questions for {combo} {arm} ({attempt + 1}/3)")
                        time.sleep(8)
                    else:
                        raise RuntimeError(f"{combo} {arm} still has unfinished questions")
                    summary = json.loads((output / "summary.json").read_text())
                    services.log(f"{combo} {arm}: {summary['correct']}/1400, {summary['elapsed_seconds']:.1f}s")
                services.stage(f"Verifying {combo}")
                archive_pair(framework, runtime, combo, expected)
            write_comparison(framework, runtime)
            services.stage("Complete: 12 online conditions verified")
        except Exception as error:
            services.stage(f"Paused: {error}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    args = parser.parse_args()
    run_matrix(args.runtime.resolve())

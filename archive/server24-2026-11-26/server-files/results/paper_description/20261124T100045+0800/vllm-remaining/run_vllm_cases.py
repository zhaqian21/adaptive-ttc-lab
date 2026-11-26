"""Run the concurrency comparison between the audited E2 and the E3 jobs."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    directory = Path(__file__).resolve().parent
    batch = directory.parent
    if sys.argv[1] == "E3" and (batch / "parallel-gpu1.ready").is_file():
        import time
        assert (batch / "E2.status").read_text().strip() == "verified"
        (batch / "gpu0-e2-release.ready").touch()
        while True:
            time.sleep(3)
    benchmark = batch / "compare-concurrency-16-32"
    if sys.argv[1] == "E3" and not (benchmark / "summary.json").is_file():
        assert (batch / "E2.status").read_text().strip() == "verified"
        benchmark.mkdir(exist_ok=True)
        (batch / "E3.status").write_text("concurrency_test\n", encoding="utf-8")
        report = batch / "compare-vllm.py"
        subprocess.run([sys.executable, str(report), str(batch)], check=True)
        with (benchmark / "queue.log").open("a", encoding="utf-8") as log:
            subprocess.run(["bash", str(directory / "benchmark_c32.sh")], stdout=log, stderr=log, check=True)
        (batch / "E3.status").write_text("running\n", encoding="utf-8")
        subprocess.run([sys.executable, str(report), str(batch)], check=True)
    implementation = directory / "run_vllm_cases_impl.py"
    os.execv(sys.executable, [sys.executable, str(implementation), *sys.argv[1:]])


if __name__ == "__main__":
    main()

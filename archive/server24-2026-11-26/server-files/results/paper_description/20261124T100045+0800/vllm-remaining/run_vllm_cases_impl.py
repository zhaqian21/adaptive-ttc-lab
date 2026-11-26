"""Dispatch the GPU 1 E4 restart to the shared work pool."""
import argparse
import os
from pathlib import Path
import sys


def main():
    scripts = Path(__file__).resolve().parent
    parallel = scripts.parent / "E4-parallel"
    if sys.argv[1] == "E4" and "--limit" not in sys.argv and (parallel / "activate").is_file():
        parser = argparse.ArgumentParser()
        parser.add_argument("case")
        parser.add_argument("--endpoint", required=True)
        parser.add_argument("--served-model", required=True)
        parser.add_argument("--vllm-version", required=True)
        parser.add_argument("--e1-config", required=True)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--workers", required=True, type=int)
        parser.add_argument("--resume", action="store_true")
        args = parser.parse_args()
        assert args.output.resolve() == scripts.parent / "E4"
        assert args.workers == 32 and args.resume
        command = [sys.executable, str(scripts / "run_e4_parallel.py"), "coordinator", "--gpu", "1",
                   "--endpoint", args.endpoint, "--served-model", args.served_model,
                   "--vllm-version", args.vllm_version]
    else:
        command = [sys.executable, str(scripts / "run_vllm_cases_parallel_base.py"), *sys.argv[1:]]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()

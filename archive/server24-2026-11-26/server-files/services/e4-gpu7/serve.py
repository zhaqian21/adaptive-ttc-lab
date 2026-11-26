import os
from contextlib import ExitStack
from pathlib import Path
import signal
import sys
import time

run_dir = Path("/srv/adaptive-ttc-lab/results/e4-gsm8k-aime26.eM54G6xe")
sys.path.insert(0, str(run_dir / "experiments/e4-gsm8k-aime26"))
from run import ENGINE_FLAGS, MODELS, released, start_service, wait_ready

here = Path(__file__).resolve().parent
model = MODELS["llama"]
engine = "/opt/inference-runtime/venv/bin/python"
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
with ExitStack() as stack:
    prm = start_service(stack, [
        sys.executable, str(run_dir / "experiments/llama-table1-825/llama_table1.py"),
        "--serve-prm", "--prm-port", "18437",
    ], os.environ.copy(), here / "prm.log")
    llm = start_service(stack, [
        engine, "-m", "vllm.entrypoints.cli.main", "serve", str(released.runner.asset_path(model)),
        "--host", "127.0.0.1", "--port", "18427", "--served-model-name", model[1], *ENGINE_FLAGS,
    ], {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, here / "engine.log")
    for process, port in ((prm, 18437), (llm, 18427)):
        wait_ready(process, port)
    print("GPU 7 Llama + PRM ready", flush=True)
    while prm.poll() is None and llm.poll() is None:
        time.sleep(2)
    raise RuntimeError(f"GPU 7 service exited: generation={llm.poll()}, PRM={prm.poll()}")


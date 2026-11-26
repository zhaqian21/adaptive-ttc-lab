"""Compare fixed K=4, fixed K=5, and dynamic K=3/5 on resident Qwen engines."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "qwen-e2-online-dynamic-3-5"
sys.path.insert(0, str(PREVIOUS))
from online_e2 import CONTEXT, ENGINE, QWEN, questions, released, start_service, wait_ready, worker
from archive import archive


LANES = {0: ("e2_fixed5", 18420, 22502), 1: ("e2_fixed4", 18421, 22503), 7: ("e2_dynamic3_to_5", 18428, 18438)}
LIMITS = {"e2_fixed5": 5, "e2_fixed4": 4, "e2_dynamic3_to_5": 5}
SERVED_NAMES = {0: "published-qwen25", 1: "published-qwen25", 7: QWEN[1]}
OPENER = build_opener(ProxyHandler({}))


def read_endpoint(port, route):
    with OPENER.open(f"http://127.0.0.1:{port}/{route}", timeout=5) as reply:
        return reply.read().decode("utf-8")


def preflight(args):
    if len(questions(args.data)["math"]) != 500:
        raise ValueError("Expected 500 frozen MATH questions")
    version = subprocess.check_output([args.engine_python, "-c",
        "from importlib.metadata import version; print(version('vllm'))"], text=True).strip()
    if version != "0.33.0":
        raise ValueError(f"Expected vLLM 0.33.0, got {version}")
    backends = {}
    for lane, (_, gen, _) in LANES.items():
        model, = json.loads(read_endpoint(gen, "v1/models"))["data"]
        if (model["id"] != SERVED_NAMES[lane] or model["max_model_len"] != CONTEXT
                or not Path(model["root"]).name.endswith(QWEN[2])):
            raise ValueError(f"GPU {lane} advertises an unexpected generation model")
        metrics = read_endpoint(gen, "metrics")
        running = re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]+\} ([\d.]+)$", metrics, re.M)
        if len(running) != 2 or any(float(n) for n in running):
            raise ValueError(f"GPU {lane} generator is processing other requests")
        sockets = subprocess.check_output(["ss", "-ltnpH", f"sport = :{gen}"], text=True)
        pids = set(re.findall(r"pid=(\d+)", sockets))
        if len(pids) != 1:
            raise ValueError(f"Cannot identify GPU {lane} generation server")
        pid = next(iter(pids))
        cmd = (Path("/proc") / pid / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if any(f"--{key}" not in cmd or cmd[cmd.index(f"--{key}") + 1] != expected
               for key, expected in ((k, f"{v:.2f}" if k == "gpu-memory-utilization" else str(v))
                                     for k, v in ENGINE.items())) or "--enable-prefix-caching" not in cmd:
            raise ValueError(f"GPU {lane} generation server does not match the experiment engine settings")
        backends[str(lane)] = {"port": gen, "model_id": model["id"], "revision": QWEN[2],
                               "settings": {**ENGINE, "enable-prefix-caching": True, "vllm": version}}
    released.write(args.output / "generation_backends.json", backends)


def coordinate(args):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != len(LANES) or len(set(devices)) != len(LANES) or not all(devices):
        raise ValueError("Expected three distinct GPU UUIDs")
    args.output.mkdir(parents=True, exist_ok=True)
    released.write(args.output / "status.json", {"phase": "starting-prm"})
    with ExitStack() as services:
        ready = []
        for (lane, (_, _, prm)), gpu in zip(LANES.items(), devices, strict=True):
            if lane == 7:
                continue
            command = [sys.executable, "-B", str(HERE.parent / "llama-table1-825/llama_table1.py"),
                       "--serve-prm", "--prm-port", str(prm)]
            proc = start_service(services, command, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                 args.output / f"gpu-{lane}-prm.log")
            ready.append((proc, prm))
        for proc, port in ready:
            wait_ready(proc, port)
        for _, _, port in LANES.values():
            model, = json.loads(read_endpoint(port, "v1/models"))["data"]
            if model["id"] != "Qwen2.5-Math-PRM-7B":
                raise ValueError("Unexpected PRM model")
        for phase in ("smoke", "formal"):
            released.write(args.output / "status.json", {"phase": phase})
            with ExitStack() as workers:
                children = {}
                for (lane, (method, _, _)), gpu in zip(LANES.items(), devices, strict=True):
                    output = args.output / phase / method
                    output.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-B", str(Path(__file__)), "--lane", str(lane),
                               "--data", str(args.data), "--output", str(output)]
                    if phase == "smoke":
                        command.append("--smoke")
                    children[method] = start_service(workers, command, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                                                     output / "worker.log")
                codes = {method: proc.wait() for method, proc in children.items()}
                if any(codes.values()):
                    raise RuntimeError(f"{phase}: {codes}")
    released.write(args.output / "status.json", {"phase": "generation-complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-python")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--lane", type=int, choices=LANES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.preflight:
        preflight(args)
    elif args.results is not None:
        archive(args.data, args.results, args.output, lanes=LANES, limits=LIMITS, reference="e2_fixed4")
    elif args.lane is not None:
        worker(args, lanes=LANES, limits=LIMITS, served_name=SERVED_NAMES[args.lane])
    else:
        coordinate(args)

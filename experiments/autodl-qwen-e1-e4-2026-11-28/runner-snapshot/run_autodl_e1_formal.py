"""Run the archived, full MATH-500 E1 protocol on AutoDL."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

root = Path(__file__).resolve().parent
source = root / "scripts/run_published_code.py"
shard = int(sys.argv[1])
assert shard in (0, 1)
archive = json.loads((root / "config/formal-runs.json").read_text())
reference = archive[f"gpu-{shard}"]["E1"]["config"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["source_sha256"]
assert reference["concurrency"] == 64 and reference["package_version"] == "0.33.0"
assert reference["sample_ids"] == list(range(shard, 500, 2))

device = json.loads((root / "backend-device.json").read_text())
runner = runpy.run_path(str(source), run_name="autodl_e1_formal")
torch = runner["torch"]
assert str(torch.__version__) == reference["runtime"]["torch"]
assert torch.version.cuda == reference["runtime"]["cuda_runtime"]
assert version("transformers") == reference["runtime"]["transformers"]
assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
assert torch.cuda.get_device_name(0) == device["device_name"] and "4090" in device["device_name"]
assert device["vllm"] == reference["package_version"]

runner["runner"].require_h20 = lambda: {
    "torch": str(torch.__version__),
    "cuda_runtime": torch.version.cuda,
    "transformers": version("transformers"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    **device,
}
args = SimpleNamespace(
    case="E1", backend="vllm", workers=reference["concurrency"], sample=None,
    seed=reference["seed"], context=32768, port=18420, model="published-qwen25",
    vllm_version=device["vllm"], output=Path(sys.argv[2]) / f"shard-{shard}",
    shards=2, shard=shard, priority_file=root / "performance/selected-profile.json",
    strict_errors=reference["strict_errors"],
)
raise SystemExit(0 if runner["run"](args) else 1)

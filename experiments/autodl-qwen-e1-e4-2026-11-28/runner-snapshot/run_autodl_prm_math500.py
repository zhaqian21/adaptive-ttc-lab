"""Replay the archived full MATH-500 E2 and E4 configuration on one 4090."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

case, output = sys.argv[1:]
assert case in {"E2", "E4"}
root = Path(__file__).resolve().parent
source = root / "scripts/run_published_code.py"
archive = json.loads((root / "config/formal-runs.json").read_text())
reference = archive["gpu-0"][case]["config"]
other = archive["gpu-1"][case]["config"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["source_sha256"]
for field in ("source_sha256", "seed", "decode", "concurrency", "strict_errors", "dataset_sha256"):
    assert reference[field] == other[field]
assert reference["concurrency"] == {"E2":48,"E4":24}[case] and reference["package_version"] == "0.33.0"

device = json.loads((root / "backend-device.json").read_text())
module = runpy.run_path(str(source), run_name="autodl_prm_formal")
torch = module["torch"]
assert str(torch.__version__) == reference["runtime"]["torch"]
assert torch.version.cuda == reference["runtime"]["cuda_runtime"]
assert version("transformers") == reference["runtime"]["transformers"]
assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
assert torch.cuda.get_device_name(0) == device["device_name"] and "4090" in device["device_name"]
assert device["vllm"] == reference["package_version"]
module["runner"].require_h20 = lambda: {
    "torch": str(torch.__version__),
    "cuda_runtime": torch.version.cuda,
    "transformers": version("transformers"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    **device,
}
args = SimpleNamespace(
    case=case, backend="vllm", workers=reference["concurrency"], sample=None, seed=reference["seed"],
    context=32768, port=18420, model="published-qwen25", vllm_version=device["vllm"],
    output=Path(output) / "math500" / "full", shards=1, shard=0,
    priority_file=root / "performance/selected-profile.json", strict_errors=reference["strict_errors"],
)
raise SystemExit(0 if module["run"](args) else 1)

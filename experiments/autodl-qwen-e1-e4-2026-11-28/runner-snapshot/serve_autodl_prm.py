"""Serve the archived BF16 PRM scorer on the local AutoDL loopback interface."""
import hashlib
from importlib.metadata import version
from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
script = root.parent.parent / "github/experiments/aime24-amo-bench-published-code-2026-11-24/scripts/serve_prm.py"
assert hashlib.sha256(script.read_bytes()).hexdigest() == "0954c5568a8a2491a3435a90a5820dd6610b8b9d269fcc7a04ac97233be13d85"
import torch

def require_autodl():
    assert torch.__version__ == "2.6.0+cu124"
    assert version("transformers") == "4.48.3"
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert "4090" in torch.cuda.get_device_name(0)

module = runpy.run_path(str(script), run_name="autodl_prm_server")
module["serve"].__globals__["require_h20"] = require_autodl
module["serve"]()

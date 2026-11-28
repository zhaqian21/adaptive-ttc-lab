#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/adaptive-ttc-lab/general-ttc
py=/root/ttc-vgpu48-runtime/e1-prm/bin/python
"$py" -u "$root/concurrent-source/run.py" --manifest "$root/full-questions.json" \
  --method E1 --workers 96 --resume --generator-model published-qwen25 \
  --code-revision ed6674002eec1b74e61d688cf7550d75d878cb88 --output "$root/full/E1"

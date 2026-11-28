#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/adaptive-ttc-lab/general-ttc
cd "$root"
sha256sum --quiet -c extra900-questions.sha256
py=/root/ttc-vgpu48-runtime/e1-prm/bin/python
"$py" -u concurrent-source/run.py --manifest "$root/extra900-questions.json" \
  --method E1 --workers 96 --generator-model published-qwen25 \
  --code-revision d19e605b76899bdc3791bec3a2b48234534166de --output "$root/full/E1-extra900"

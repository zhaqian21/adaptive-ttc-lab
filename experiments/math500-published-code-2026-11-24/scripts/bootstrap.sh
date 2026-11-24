#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
upstream_commit=d9ca5f71f5c7d155153e8f7decd8b1c524968775
cd "$root"

if [[ ! -d upstream/.git ]]; then
  git clone https://github.com/AhsanBilal7/adaptive-test-time-compute.git upstream
  git -C upstream checkout --detach "$upstream_commit"
fi
[[ "$(git -C upstream rev-parse HEAD)" == "$upstream_commit" ]]
[[ -z "$(git -C upstream status --porcelain --untracked-files=no)" ]]

uv_bin="${UV_BIN:-uv}"
python="${TTC_PYTHON:-python3.12}"
if [[ ! -x .venv/bin/python ]]; then
  "$uv_bin" venv --python "$python" .venv
fi
if [[ -s environment.lock.txt ]]; then
  "$uv_bin" pip sync --python .venv/bin/python environment.lock.txt
else
  "$uv_bin" pip install --python .venv/bin/python \
    -r upstream/requirements.txt -c scripts/constraints.txt \
    antlr4-python3-runtime pyparsing
  "$uv_bin" pip freeze --python .venv/bin/python > environment.lock.txt
fi
"$uv_bin" pip check --python .venv/bin/python

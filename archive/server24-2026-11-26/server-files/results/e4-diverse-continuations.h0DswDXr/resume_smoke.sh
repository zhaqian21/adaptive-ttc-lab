#!/usr/bin/env bash
set -euo pipefail
cd '/srv/adaptive-ttc-lab/results/e4-diverse-continuations.h0DswDXr'
until [[ -s run.exit ]]; do sleep 5; done
[[ "$(cat run.exit)" != 0 && "$(cat service-restoration)" == restored ]]
mkdir -- initial-smoke
mv -- results driver.log run.exit service-restoration status initial-smoke/
entry="$PWD/experiments/e4-diverse-continuations"
mv -- "$entry/run.py.next" "$entry/run.py"
export PYTHONPATH="$entry:$PWD/experiments/aime24-amo-bench-published-code-2026-11-24/scripts:$PWD/source/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
env -u TTC_E4_STUDY source/.venv/bin/python -B - <<'PY'
from types import SimpleNamespace
from unittest.mock import patch
import run as e
s=e.MeteredSession(5)
e.next_stage(s, 6, 0, 123, "")
request=lambda: s.post("http://127.0.0.1:22500/v1/completions",json={"prompt":[1,2,3],"max_tokens":4})
client=SimpleNamespace(max_retries=5, delay=0)
with patch("requests.sessions.Session.post") as sender:
    try:
        e.released.TransformerWrapper.execute_with_retries(client, request)
    except e.BudgetReached:
        pass
    else:
        raise AssertionError("Budget rejection should cross upstream transport retries")
    assert not sender.called and s.current["denied_tokens"]==7 and s.charged==0
print("Budget rejection passed through the upstream retry boundary.")
PY
exec env -u TTC_E4_STUDY bash "$PWD/experiments/llama-table1-825/run_24.sh" "$PWD" \
  /opt/inference-runtime/venv/bin/python "$entry/run.py" >driver.log 2>&1

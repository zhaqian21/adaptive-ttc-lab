#!/usr/bin/env bash
set -euo pipefail
tuning="$(cd "$(dirname "$0")" && pwd)"
batch="$(cd "$tuning/../.." && pwd)"
py="$batch/source/.venv/bin/python"
cd "$batch"
exec 7>"$tuning/.trial.lock"
flock -n 7
for ((second=0; second<1000; second+=10)); do
  [[ "$(cat E4.status)" == running ]] || { echo "E4 has left the running state"; exit 1; }
  tmux has-session -t adaptive-ttc-e4-gpu0 || { echo "GPU 0 trial ended before the measurement window"; exit 1; }
  committed="$(grep -c "^GPU 0 committed" "$tuning/gpu0-c64-trial/evaluator.log" 2>/dev/null || true)"
  [[ "${committed:-0}" -lt 3 ]] || break
  sleep 10
done
[[ "${committed:-0}" -ge 3 ]] || { echo "GPU 0 did not reach three checkpoints in time for a reliable sample"; exit 1; }
echo "$(date -Iseconds) Measuring five minutes of stable E4 throughput."
"$py" "$tuning/collect_e4_speed.py" trial-64-32 --seconds 300
export PYTHONPATH="$batch/source/scripts"
"$py" "$tuning/evaluate_c64.py"
if "$py" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["promote_gpu1"] else 1)' "$tuning/decision.json"; then
  [[ "$(cat E4.status)" == running ]]
  bash "$batch/switch-e4-tuned.sh" 1 3309483 64 gpu1-c64-promoted
else
  echo "GPU 1 stays at 32."
fi

#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/adaptive-ttc-lab/general-ttc
cd "$root"
py=/root/ttc-vgpu48-runtime/e1-prm/bin/python
printf 'Starting VersaPRM with a larger connection queue\n'
for n in {1..60}; do
    state="$(ps -p 3057 -o stat= || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then break; fi
    sleep 1
done
test -z "$state" || [[ "$state" == Z* ]]
PYTHONPATH="$root/python-deps" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nohup "$py" -u concurrent-source/serve_prm.py --models "$root/models" </dev/null >logs/versaprm-backlog.log 2>&1 &
prm_pid=$!
printf '%s\n' "$prm_pid" >logs/versaprm.pid
for n in {1..100}; do
    curl -fsS --max-time 3 http://127.0.0.1:18423/health >/dev/null 2>&1 && break
    kill -0 "$prm_pid"
    sleep 3
done
curl -fsS --max-time 8 -H 'Content-Type: application/json' --data '{"question":"What is 2+3?","steps":["Add two and three to get five."]}' http://127.0.0.1:18423/score
printf '\n'
export E2_WORKERS=96
exec bash queue-e3-e2-e4/run.sh --resume

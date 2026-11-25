#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 || $# == 3 ]] || { echo "usage: $0 RUN_DIRECTORY VLLM_PYTHON [EVALUATOR]" >&2; exit 2; }
here="$(cd -- "$(dirname -- "$0")" && pwd)"
root="$(realpath -- "$1")"
engine="$(realpath -s -- "$2")"
evaluator="$(realpath -- "${3:-$here/evaluate.py}")"
[[ -f "$evaluator" ]]
services=(ttc-published-service-0 ttc-published-service-1 ttc-published-prm-gpu0)
ports=(18420 18421 18422)
commands=()
need_restore=0

exec 9>"$root/run.lock"
flock -n 9

session_exists() {
    tmux has-session -t "=$1" 2>/dev/null
}

wait_port() {
    local port=$1
    for ((attempt=0;attempt<120;attempt++)); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        sleep 2
    done
    return 1
}

restore() {
    local rc=$? restored=1
    trap - EXIT
    if (( need_restore )); then
        for lane in 0 1 2; do
            if ! session_exists "${services[lane]}"; then
                tmux new-session -d -s "${services[lane]}" "${commands[lane]}" || restored=0
            fi
            wait_port "${ports[lane]}" || restored=0
        done
        if (( restored )); then
            echo restored > "$root/service-restoration"
        else
            echo failed > "$root/service-restoration"
            (( rc != 0 )) || rc=1
        fi
    fi
    printf '%s\n' "$rc" > "$root/run.exit"
    exit "$rc"
}
trap restore EXIT
trap 'exit 143' TERM INT

echo preflight > "$root/status"
for port in 22500 22501 22502 22503; do [[ -z "$(ss -ltnH "sport = :$port")" ]]; done
for name in "${services[@]}"; do
    session_exists "$name"
    command="$(tmux display-message -p -t "$name:" '#{pane_start_command}')"
    [[ -n "$command" ]]
    commands+=("$command")
done
session_exists ttc-e4-gpu7
for port in 18420 18421 18427; do
    curl -fsS --max-time 5 "http://127.0.0.1:$port/metrics" |
        awk '/^vllm:num_requests_(running|waiting)\{/ {seen++; total+=$NF} END {exit !(seen>=2 && total==0)}'
done
curl -fsS --max-time 5 http://127.0.0.1:18437/health >/dev/null

export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0,1,7 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^GPU-[^,]+,GPU-[^,]+,GPU-[^,]+$ ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$here/../aime24-amo-bench-published-code-2026-11-24/scripts:$root/source/scripts"

need_restore=1
echo switching-models > "$root/status"
for lane in 2 1 0; do tmux send-keys -t "${services[lane]}:" C-c; done
for ((attempt=0;attempt<60;attempt++)); do
    occupied="$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -z "$occupied" ]]; then break; fi
    sleep 2
done
[[ -z "$occupied" ]]

echo running > "$root/status"
"$root/source/.venv/bin/python" -B "$evaluator" --data "$root/data" \
    --output "$root/results" --engine-python "$engine"
echo generation-complete > "$root/status"

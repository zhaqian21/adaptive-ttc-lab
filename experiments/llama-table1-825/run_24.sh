#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: $0 PREPARED_OUTPUT VLLM_PYTHON" >&2
    exit 2
fi
entry="$(cd -- "$(dirname -- "$0")" && pwd)"
root="$(realpath -- "$1")"
engine="$(realpath -- "$2")"
source="$root/source"
release="$entry/../aime24-amo-bench-published-code-2026-11-24"
services=(ttc-published-service-0 ttc-published-service-1 ttc-published-prm-gpu0)
ports=(18420 18421 18422)
commands=()
need_restore=0

wait_port() {
    local port=$1
    for ((retry=0;retry<90;retry++)); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        sleep 2
    done
    return 1
}

has_service_session() {
    tmux list-sessions -F '#S' 2>/dev/null | grep -Fx "$1" >/dev/null
}

restore() {
    local rc=$?
    trap - EXIT
    if (( need_restore )); then
        local restore_ok=1
        for lane in 0 1 2; do
            if ! has_service_session "${services[lane]}"; then
                tmux new-session -d -s "${services[lane]}" "${commands[lane]}" || restore_ok=0
            fi
            wait_port "${ports[lane]}" || restore_ok=0
        done
        if (( restore_ok )); then
            printf '%s\n' restored > "$root/service-restoration"
        else
            printf '%s\n' failed > "$root/service-restoration"
            (( rc != 0 )) || rc=1
        fi
    fi
    printf '%s\n' "$rc" > "$root/run.exit"
    exit "$rc"
}
trap restore EXIT
trap 'exit 143' TERM INT

printf '%s\n' waiting-for-model-assets > "$root/status"
until [[ -s "$root/assets.exit" ]]; do sleep 15; done
[[ "$(cat "$root/assets.exit")" == 0 ]]
"$engine" -c 'from importlib.metadata import version; assert version("vllm") == "0.33.0"'
for port in 22500 22501 22502 22503; do
    listeners="$(ss -ltnH "sport = :$port")"
    [[ -z "$listeners" ]] || { echo "Port $port already has a listener" >&2; exit 1; }
done
for name in "${services[@]}"; do
    has_service_session "$name"
    command="$(tmux display-message -p -t "$name:" '#{pane_start_command}')"
    [[ -n "$command" ]]
    commands+=("$command")
done
for port in 18420 18421; do
    curl -fsS --max-time 5 "http://127.0.0.1:$port/metrics" |
        awk '/^vllm:num_requests_(running|waiting)\{/ {seen++; total+=$NF} END {exit !(seen>=2 && total==0)}'
done
need_restore=1
printf '%s\n' switching-models > "$root/status"
for lane in 2 1 0; do
    tmux send-keys -t "${services[lane]}:" C-c
done
for ((retry=0;retry<60;retry++)); do
    occupied="$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -z "$occupied" ]]; then break; fi
    sleep 2
done
[[ -z "$occupied" ]]
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0,1 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^GPU-[^,]+,GPU-[^,]+$ ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$release/scripts:$source/scripts"
printf '%s\n' evaluating > "$root/status"
env -u SLURM_GPUS_ON_NODE -u SLURM_JOB_ID "$source/.venv/bin/python" "$entry/llama_table1.py" \
    --manifest "$source/datasets/manifest.json" --output "$root/results" --engine-python "$engine"
printf '%s\n' verified > "$root/status"

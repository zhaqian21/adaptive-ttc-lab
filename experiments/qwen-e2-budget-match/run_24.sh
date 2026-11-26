#!/usr/bin/env bash
set -euo pipefail

model=qwen
if [[ "${1:-}" == --llama ]]; then model=llama; shift; fi
[[ $# -ge 2 ]] || { echo "usage: $0 [--llama] RUN_DIRECTORY VLLM_PYTHON [DATASET ...]" >&2; exit 2; }
here="$(cd -- "$(dirname -- "$0")" && pwd)"
root="$(realpath -- "$1")"
engine="$(realpath -s -- "$2")"
shift 2
datasets=()
if (( $# )); then datasets=(--datasets "$@"); fi
python="$root/source/.venv/bin/python"
services=(ttc-published-prm-gpu0)
ports=(18422)
new_ports=(22502 22503)
if [[ "$model" == llama ]]; then
    services=(ttc-published-service-0 ttc-published-service-1 ttc-qwen-gpu7 "$services")
    ports=(18420 18421 18428 "$ports")
    new_ports+=(22500 22501 22504)
fi
commands=()
directories=()
pids=()
restore_services=0
exec 9>"$root/run.lock"
flock -n 9

phase() {
    printf "%s\n" "$1" > "$root/status"
}
has_session() {
    tmux has-session -t "=$1" 2>/dev/null
}
healthy() {
    curl -fsS --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}
listener_pid() {
    ss -ltnpH "sport = :$1" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
}
restore() {
    local rc=$? restored=1 ready
    trap - EXIT
    if (( restore_services )); then
        for i in "${!services[@]}"; do
            if [[ "${ports[i]}" == 18422 ]]; then continue; fi
            if ! has_session "${services[i]}"; then
                tmux new-session -d -s "${services[i]}" -c "${directories[i]}" "${commands[i]}" || restored=0
            fi
        done
        for i in "${!services[@]}"; do
            if [[ "${ports[i]}" == 18422 ]] && ! has_session "${services[i]}"; then
                tmux new-session -d -s "${services[i]}" -c "${directories[i]}" "${commands[i]}" || restored=0
            fi
            ready=0
            for ((attempt=0;attempt<150;attempt++)); do
                if healthy "${ports[i]}"; then ready=1; break; fi
                has_session "${services[i]}" || break
                sleep 2
            done
            if (( !ready )); then
                printf 'Service %s did not recover on port %s\n' "${services[i]}" "${ports[i]}" >&2
                restored=0
            fi
        done
        if (( restored )); then
            echo restored > "$root/service-restoration"
        else
            echo failed > "$root/service-restoration"
            rc=1
        fi
    fi
    if (( rc == 0 )); then phase complete; else phase failed; fi
    printf '%s\n' "$rc" > "$root/run.exit"
    exit "$rc"
}
trap restore EXIT
trap 'printf "Experiment launcher failed on line %s\n" "$LINENO" >&2' ERR
trap 'exit 143' TERM INT

phase preflight
rm -f -- "$root/run.exit" "$root/service-restoration"
mkdir -p -- "$root/results"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0,1,7 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^GPU-[^,]+,GPU-[^,]+,GPU-[^,]+$ ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$here/../aime24-amo-bench-published-code-2026-11-24/scripts:$root/source/scripts"
"$python" -B "$here/compare.py" --data "$root/data" --output "$root/results" --engine-python "$engine" \
    --preflight --model "$model" "${datasets[@]}"
for port in "${new_ports[@]}"; do [[ -z "$(ss -ltnH "sport = :$port")" ]]; done
healthy 18438
prm7_pid="$(listener_pid 18438)"
[[ "$prm7_pid" =~ ^[1-9][0-9]*$ ]]
for i in "${!services[@]}"; do
    session="${services[i]}"
    port="${ports[i]}"
    has_session "$session"
    healthy "$port"
    if [[ "$port" != 18422 ]]; then
        curl -fsS --max-time 3 "http://127.0.0.1:$port/metrics" |
            awk '/^vllm:num_requests_(running|waiting)\{/ {n++; total+=$NF} END {exit !(n==2 && total==0)}'
    fi
    command="$(tmux display-message -p -t "$session:" '#{pane_start_command}')"
    directory="$(tmux display-message -p -t "$session:" '#{pane_current_path}')"
    pane="$(tmux display-message -p -t "$session:" '#{pane_pid}')"
    pid="$(listener_pid "$port")"
    [[ -n "$command" && -d "$directory" && "$pid" =~ ^[1-9][0-9]*$ ]]
    parent="$(ps -o ppid= -p "$pid" | tr -d ' ')"
    [[ "$pid" == "$pane" || "$parent" == "$pane" ]]
    commands+=("$command")
    directories+=("$directory")
    pids+=("$pid")
done

released() {
    local i active
    for i in "${!services[@]}"; do
        if has_session "${services[i]}" || [[ -d "/proc/${pids[i]}" ]]; then return 1; fi
    done
    if [[ "$model" == llama ]]; then
        active="$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
        [[ -z "$active" ]] || return 1
        active="$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
        [[ "$active" == "$prm7_pid" ]] || return 1
    fi
}
phase switching-services
restore_services=1
for session in "${services[@]}"; do tmux send-keys -t "$session:" C-c; done
for ((i=0;i<60;i++)); do
    if released; then break; fi
    sleep 1
done
if ! released; then
    echo "Previous services have not released their assigned GPUs" >&2
    exit 1
fi

phase running
"$python" -B "$here/compare.py" --data "$root/data" --output "$root/results" --engine-python "$engine" \
    --model "$model" "${datasets[@]}"
phase auditing
"$python" -B "$here/compare.py" --data "$root/data" --results "$root/results" --output "$root/export" \
    --model "$model" "${datasets[@]}"
cp -- "$root/results/generation_backends.json" "$root/export/generation_backends.json"
tar -cf "$root/budget-match-review.tar.tmp" -C "$root/export" .
mv -- "$root/budget-match-review.tar.tmp" "$root/budget-match-review.tar"
touch "$root/audit.ok"
phase restoring-services

#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || { echo "usage: $0 RUN_DIRECTORY VLLM_PYTHON" >&2; exit 2; }
here="$(cd -- "$(dirname -- "$0")" && pwd)"
root="$(realpath -- "$1")"
engine="$(realpath -s -- "$2")"
python="$root/source/.venv/bin/python"
old=ttc-published-prm-gpu0
old_port=18422
restore_prm=0
exec 9>"$root/run.lock"
flock -n 9

phase() {
    printf "%s\n" "$1" > "$root/status"
}
has_session() {
    tmux has-session -t "=$1" 2>/dev/null
}
restore() {
    local rc=$?
    trap - EXIT
    if (( restore_prm )); then
        if ! has_session "$old"; then
            tmux new-session -d -s "$old" -c "$old_dir" "$old_command" || rc=1
        fi
        local ready=0
        for ((i=0;i<120;i++)); do
            if curl -fsS --max-time 2 "http://127.0.0.1:$old_port/health" >/dev/null; then
                ready=1
                break
            fi
            has_session "$old" || break
            sleep 2
        done 2>/dev/null
        if (( ready )); then
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
trap 'exit 143' TERM INT

phase preflight
rm -f -- "$root/run.exit"
mkdir -p -- "$root/results"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0,1,7 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^GPU-[^,]+,GPU-[^,]+,GPU-[^,]+$ ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export PYTHONPATH="$here/../aime24-amo-bench-published-code-2026-11-24/scripts:$root/source/scripts"
"$python" -B "$here/compare.py" --data "$root/data" --output "$root/results" --engine-python "$engine" --preflight
[[ -z "$(ss -ltnH '( sport = :22502 or sport = :22503 )')" ]]
curl -fsS --max-time 3 http://127.0.0.1:18438/health >/dev/null
curl -fsS --max-time 3 "http://127.0.0.1:$old_port/health" >/dev/null
has_session "$old"
old_command="$(tmux display-message -p -t "$old:" '#{pane_start_command}')"
old_dir="$(tmux display-message -p -t "$old:" '#{pane_current_path}')"
old_pid="$(tmux display-message -p -t "$old:" '#{pane_pid}')"
[[ -n "$old_command" && -d "$old_dir" && "$old_pid" =~ ^[1-9][0-9]*$ ]]
ss -ltnp "sport = :$old_port" | awk -v pid="$old_pid" 'index($0,"pid="pid","){seen=1} END{exit !seen}'
phase switching-prm
restore_prm=1
tmux send-keys -t "$old:" C-c
for ((i=0;i<40;i++)); do
    if ! has_session "$old" && [[ ! -d "/proc/$old_pid" ]]; then break; fi
    sleep 1
done
! has_session "$old"
[[ ! -d "/proc/$old_pid" ]]

phase running
"$python" -B "$here/compare.py" --data "$root/data" --output "$root/results"
phase auditing
"$python" -B "$here/compare.py" --data "$root/data" --results "$root/results" --output "$root/export"
cp -- "$root/results/generation_backends.json" "$root/export/generation_backends.json"
tar -cf "$root/budget-match-review.tar.tmp" -C "$root/export" .
mv -- "$root/budget-match-review.tar.tmp" "$root/budget-match-review.tar"
touch "$root/audit.ok"
phase restoring-prm

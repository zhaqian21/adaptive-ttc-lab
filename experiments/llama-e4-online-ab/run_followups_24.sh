#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: $0 PREVIOUS_RUN_OR_QUEUE PREPARED_QUEUE VLLM_PYTHON [STUDY ...]" >&2
    exit 2
fi

previous="$(realpath -- "$1")"
queue="$(realpath -- "$2")"
engine="$(realpath -s -- "$3")"
shift 3
if [[ $# == 0 ]]; then
    set -- llama-amc23 qwen-math500
fi
entry="$(cd -- "$(dirname -- "$0")" && pwd)"
launcher="$entry/../llama-table1-825/run_24.sh"

exec 9>"$queue/queue.lock"
flock -n 9
trap 'rc=$?; printf "%s\n" "$rc" > "$queue/queue.exit"' EXIT
trap 'exit 143' TERM INT

status() {
    printf '%s\n' "$1" > "$queue/queue-status.tmp"
    mv -- "$queue/queue-status.tmp" "$queue/queue-status"
}

if [[ -f "$previous/queue-status" ]]; then
    status waiting-for-previous-queue
    until [[ -s "$previous/queue.exit" ]]; do sleep 15; done
    [[ "$(cat "$previous/queue.exit")" == 0 && "$(cat "$previous/queue-status")" == verified ]]
else
    status waiting-for-llama-math500
    until [[ -s "$previous/run.exit" ]]; do sleep 15; done
    [[ "$(cat "$previous/run.exit")" == 0 && "$(cat "$previous/service-restoration")" == restored ]]
fi

for study in "$@"; do
    case "$study" in
        llama-amc23|qwen-amc23) plan=llama_amc23_plan.json ;;
        qwen-math500) plan=qwen_math500_plan.json ;;
        *) echo "Unknown study: $study" >&2; exit 2 ;;
    esac
    target="$queue/$study"
    status "$study"
    if [[ ! -s "$target/run.exit" ]] || [[ "$(cat "$target/run.exit")" != 0 ]]; then
        TTC_E4_STUDY="$study" bash "$launcher" "$target" "$engine" "$entry/online_ab.py" > "$target/driver.log" 2>&1
    fi
    [[ "$(cat "$target/service-restoration")" == restored ]]
    "$target/source/.venv/bin/python" -B "$entry/compare_online.py" --root "$target/results" \
        --output "$target/results/comparison" --plan "$entry/$plan"
done
status verified

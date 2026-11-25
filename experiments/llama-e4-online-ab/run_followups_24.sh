#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: $0 CURRENT_LLAMA_RUN PREPARED_QUEUE VLLM_PYTHON" >&2
    exit 2
fi

previous="$(realpath -- "$1")"
queue="$(realpath -- "$2")"
engine="$(realpath -s -- "$3")"
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

status waiting-for-llama-math500
until [[ -s "$previous/run.exit" ]]; do sleep 15; done
[[ "$(cat "$previous/run.exit")" == 0 && "$(cat "$previous/service-restoration")" == restored ]]

for study in llama-amc23 qwen-math500; do
    target="$queue/$study"
    status "$study"
    if [[ ! -s "$target/run.exit" ]] || [[ "$(cat "$target/run.exit")" != 0 ]]; then
        TTC_E4_STUDY="$study" bash "$launcher" "$target" "$engine" "$entry/online_ab.py" > "$target/driver.log" 2>&1
    fi
    [[ "$(cat "$target/service-restoration")" == restored ]]
    "$target/source/.venv/bin/python" -B "$entry/compare_online.py" --root "$target/results" \
        --output "$target/results/comparison" --plan "$entry/${study//-/_}_plan.json"
done
status verified

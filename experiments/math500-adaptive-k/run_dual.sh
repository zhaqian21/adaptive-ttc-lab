#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 || ( $4 != paired && $4 != adaptive ) ]]; then
    echo "usage: $0 SOURCE BASELINE_SCRIPTS OUTPUT {paired|adaptive} [SAMPLE]" >&2
    exit 2
fi

entry="$(cd -- "$(dirname -- "$0")" && pwd)"
source_root="$(realpath -- "$1")"
baseline_scripts="$(realpath -- "$2")"
mkdir -p -- "$3"
output="$(realpath -- "$3")"
mode=$4
python="$source_root/.venv/bin/python"
workers=24
sample=()
if [[ $# == 5 ]]; then
    workers=8
    sample=(--sample "$5")
fi

export PYTHONPATH="$entry/../math500-adaptive-revision:$baseline_scripts:$source_root/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for port in 18420 18421 18422; do
    curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/health" >/dev/null
done
for port in 18420 18421; do
    curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/version" |
        "$python" -c 'import json,sys; assert json.load(sys.stdin)["version"] == "0.33.0"'
done

SECONDS=0
pids=()
for lane in 0 1; do
    (
        shard="$output/shard-$lane"
        args=(--mode "$mode" --port "$((18420 + lane))" --shard "$lane" --workers "$workers"
              --output "$shard" "${sample[@]}")
        if ! "$python" "$entry/run_adaptive_k.py" "${args[@]}"; then
            if [[ -f "$shard/execution.json" && ! -f "$shard/first-attempt.json" ]]; then
                cp -- "$shard/execution.json" "$shard/first-attempt.json"
            fi
            "$python" "$entry/run_adaptive_k.py" "${args[@]}"
        fi
    ) >> "$output/shard-$lane.log" 2>&1 &
    pids+=("$!")
done
trap 'kill -TERM "${pids[@]}" 2>/dev/null || true; wait || true; exit 143' INT TERM

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]]
"$python" "$entry/audit.py" "$output" --wall-seconds "$SECONDS"

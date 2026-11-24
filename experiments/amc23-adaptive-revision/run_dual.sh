#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 || $# == 5 ]] || {
    echo "usage: $0 SOURCE BASELINE_SCRIPTS AMC23_PARQUET OUTPUT [SAMPLE_PER_SHARD]" >&2
    exit 2
}
entry="$(cd -- "$(dirname -- "$0")" && pwd)"
source_root="$(realpath -- "$1")"
baseline="$(realpath -- "$2")"
dataset="$(realpath -- "$3")"
mkdir -p -- "$4"
output="$(realpath -- "$4")"
python="$source_root/.venv/bin/python"
workers=16
sample=()
if [[ $# == 5 ]]; then
    workers=$5
    sample=(--sample-per-shard "$5")
fi

export PYTHONPATH="$entry/../math500-adaptive-revision:$baseline:$source_root/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for port in 18420 18421 18422; do
    curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/health" >/dev/null
done
for port in 18420 18421; do
    curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$port/version" |
      "$python" -c 'import json,sys; assert json.load(sys.stdin)["version"] == "0.33.0"'
done

pids=()
for lane in 0 1; do
    (
        shard="$output/shard-$lane"
        args=(--dataset-file "$dataset" --output "$shard" --port "$((18420+lane))" --shard "$lane"
              --workers "$workers" "${sample[@]}")
        if ! "$python" "$entry/run_amc23.py" "${args[@]}"; then
            if [[ -f "$shard/execution.json" ]]; then
                cp -- "$shard/execution.json" "$shard/first-attempt.json"
            fi
            "$python" "$entry/run_amc23.py" "${args[@]}"
        fi
    ) >> "$output/shard-$lane.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]]
"$python" "$entry/run_amc23.py" --audit --dataset-file "$dataset" --output "$output" "${sample[@]}"

#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 6 ]]; then
    echo "usage: $0 SOURCE BASELINE_SCRIPTS PRIOR_RUN OUTPUT CUTOFF_ISO LLAMA_LAUNCHER" >&2
    exit 2
fi
entry="$(cd -- "$(dirname -- "$0")" && pwd)"
source_root="$(realpath -- "$1")"
baseline="$(realpath -- "$2")"
prior="$(realpath -- "$3")"
mkdir -p -- "$4"
output="$(realpath -- "$4")"
cutoff="$(date --date="$5" +%s)"
llama=$6
trap 'rc=$?; printf "%s\n" "$rc" > "$output/driver.exit"' EXIT

printf '%s\n' waiting-for-online-comparison > "$output/status"
until [[ -s "$prior/driver.exit" ]]; do sleep 20; done
[[ "$(cat "$prior/driver.exit")" == 0 && "$(cat "$prior/status")" == verified ]]
[[ -s "$prior/full/comparison/summary.json" ]]

for method in factorial g f fg; do
    printf '%s\n' "$method-smoke" > "$output/status"
    bash "$entry/run_dual.sh" "$source_root" "$baseline" "$output/smoke/$method" "$method" 8
    printf '%s\n' "$method-formal" > "$output/status"
    bash "$entry/run_dual.sh" "$source_root" "$baseline" "$output/full/$method" "$method"
done
printf '%s\n' matched > "$output/status"
export PYTHONPATH="$entry/../math500-adaptive-k:$entry/../math500-adaptive-revision:$entry/../math500-prm-answer-aggregation"
"$source_root/.venv/bin/python" "$entry/matched.py" "$output/full"
date -Is > "$output/modules.completed"
if (( $(date +%s) < cutoff )); then
    printf '%s\n' llama > "$output/status"
    bash "$llama"
else
    printf '%s\n' cutoff-reached > "$output/llama.status"
fi
printf '%s\n' verified > "$output/status"

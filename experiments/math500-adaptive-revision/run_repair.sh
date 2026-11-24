#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 6 ]]; then
    echo "usage: $0 SOURCE BASELINE_SCRIPTS BASELINE_E4_JSONL FIRST_PASS REPAIR_OUTPUT VLLM_VERSION" >&2
    exit 2
fi

entry="$(cd -- "$(dirname -- "$0")" && pwd)"
source_root="$(cd -- "$1" && pwd)"
baseline_scripts="$(cd -- "$2" && pwd)"
reference="$(realpath -- "$3")"
first_pass="$(cd -- "$4" && pwd)"
mkdir -p -- "$5"
repair="$(cd -- "$5" && pwd)"
[[ "$repair" != "$first_pass" ]]
version=$6
python="$source_root/.venv/bin/python"

export PYTHONPATH="$baseline_scripts:$source_root/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0

for lane in 0 1; do
    failed_rows="$("$python" -c 'import json, sys; print(*sorted(map(int, json.load(open(sys.argv[1]))["failures"])))' \
        "$first_pass/shard-$lane/execution.json")"
    [[ -n "$failed_rows" ]] || continue
    read -ra rows <<< "$failed_rows"
    if [[ -f "$repair/shard-$lane/execution.json" ]] && "$python" -c '
import json, sys
from pathlib import Path
directory = Path(sys.argv[1])
execution = json.loads((directory / "execution.json").read_text())
expected = {int(row) for row in sys.argv[2:]}
actual = {int(path.stem) for path in (directory / "problems").glob("*.json")}
sys.exit(bool(execution["failures"]) or execution["complete"] != len(expected) or actual != expected)
' "$repair/shard-$lane" "${rows[@]}"; then
        continue
    fi
    subset=()
    for row in "${rows[@]}"; do subset+=(--only-row "$row"); done
    "$python" "$entry/run_math500.py" --port "$((18420 + lane))" --prm-port 18422 \
        --vllm-version "$version" --workers 24 --seed 0 --token-budget 125000 --initial-roots 3 \
        --shards 2 --shard "$lane" "${subset[@]}" --output "$repair/shard-$lane" \
        > "$repair/shard-$lane.log" 2>&1
done

"$python" "$entry/audit_math500.py" "$first_pass" "$reference" "$repair/export" --repair-run "$repair"

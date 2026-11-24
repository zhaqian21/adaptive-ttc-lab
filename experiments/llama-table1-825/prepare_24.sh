#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 BASELINE_SCRIPTS QWEN_SOURCE AIME_AMO_DATA_DIR OUTPUT" >&2
    exit 2
fi
entry="$(cd -- "$(dirname -- "$0")" && pwd)"
baseline="$(realpath -- "$1")"
qwen="$(realpath -- "$2")"
datasets="$(realpath -- "$3")"
mkdir -p -- "$4"
root="$(realpath -- "$4")"
source="$root/source"
release="$entry/../aime24-amo-bench-published-code-2026-11-24"
mkdir -p "$source/scripts" "$source/datasets" "$root/logs"
trap 'rc=$?; printf "%s\n" "$rc" > "$root/assets.exit"' EXIT
cp -- "$baseline/"*.py "$source/scripts/"
cp -- "$release/config/datasets.json" "$source/datasets/manifest.json"
[[ -e "$source/upstream" ]] || ln -s "$qwen/upstream" "$source/upstream"
[[ -e "$source/.venv" ]] || ln -s "$qwen/.venv" "$source/.venv"

"$source/.venv/bin/python" - "$entry/assets.json" "$qwen" "$datasets" "$source" <<'PY'
import json
from pathlib import Path
import sys

manifest, qwen, datasets, source = map(Path, sys.argv[1:])
for entry in json.loads(manifest.read_bytes()):
    relative = Path(entry["path"])
    dest = source / relative
    if not dest.exists():
        existing = qwen / relative
        if relative.parts[0] == "datasets":
            existing = datasets / relative.name
        if existing.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(existing.resolve())
PY
"$source/.venv/bin/python" "$entry/prepare_assets.py" "$source" "$entry/assets.json"

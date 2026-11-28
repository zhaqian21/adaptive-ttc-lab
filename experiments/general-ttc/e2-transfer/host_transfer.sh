#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: host_transfer.sh RUN_DIRECTORY" >&2
    exit 2
fi
runtime="$(realpath -- "$1")"
here="$(cd -- "$(dirname -- "$0")" && pwd)"
python=/root/ttc-vgpu48-runtime/e1-prm/bin/python
exec 9>"$runtime/host.lock"
flock -n 9

if [[ ! -s "$runtime/code-revision" ]]; then
    printf 'Preparing GitHub branch\n' >"$runtime/stage"
    "$python" -B "$here/publish_transfer.py" initialize --runtime "$runtime" >>"$runtime/publication.log" 2>&1
fi
if [[ ! -f "$runtime/run.exit" || "$(cat "$runtime/run.exit")" != 0 ]]; then
    status=0
    "$python" -B -u "$here/run_transfer.py" --runtime "$runtime" >>"$runtime/queue.log" 2>&1 || status=$?
    printf '%s\n' "$status" >"$runtime/run.exit"
    if (( status )); then
        exit "$status"
    fi
fi
printf 'Publishing verified results\n' >"$runtime/stage"
for attempt in 1 2 3 4; do
    if "$python" -B -u "$here/publish_transfer.py" publish --runtime "$runtime" >>"$runtime/publication.log" 2>&1; then
        printf 'Complete: results published in %s\n' "$(cat "$runtime/pr-url")" >"$runtime/stage"
        exit 0
    fi
    printf 'GitHub publication attempt %s failed\n' "$attempt" >>"$runtime/publication.log"
    sleep "$((30 * attempt))"
done
printf 'All experiments completed. GitHub publication requires a retry.\n' >"$runtime/stage"
exit 1

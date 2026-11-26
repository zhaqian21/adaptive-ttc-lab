#!/usr/bin/env bash
set -euo pipefail
batch="$(cd "$(dirname "$0")" && pwd)"
cd "$batch"
while [[ ! -f gpu0-e2-release.ready ]]; do
  [[ "$(cat E2.status)" != failed ]] || exit 1
  sleep 2
done
[[ "$(cat E2.status)" == verified && -s E2/audit.json ]]
pane="$(tmux list-panes -t adaptive-ttc-paper-vllm -F "#{pane_pid}")"
[[ "$pane" == 560109 && "$(ps -o pgid= -p "$pane" | xargs)" == "$pane" ]]
[[ "$(ps -o uid= -p "$pane" | xargs)" == "$(id -u)" ]]
echo "$(date -Iseconds) E2 audited; retiring the old GPU 0 queue at its E3 gate."
kill -TERM -- "-$pane"
for ((second=0; second<40; second++)); do
  [[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]] && break
  sleep 1
done
[[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]
exec bash "$batch/vllm-remaining/run_e3_gpu0.sh"

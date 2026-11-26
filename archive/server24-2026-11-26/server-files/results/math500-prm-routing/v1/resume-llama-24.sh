#!/usr/bin/env bash
set -euo pipefail
route=$1
launcher=$2
trap 'rc=$?; printf "%s\n" "$rc" > "$route/driver.exit"' EXIT
printf '%s\n' llama > "$route/status"
bash "$launcher"
printf '%s\n' verified > "$route/status"

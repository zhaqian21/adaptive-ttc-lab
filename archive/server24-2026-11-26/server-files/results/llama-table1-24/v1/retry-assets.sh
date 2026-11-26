#!/usr/bin/env bash
set -uo pipefail
entry=$1
baseline=$2
source=$3
datasets=$4
out=$5
for delay in 5 20 45 90; do
    printf '\nExact asset retry at %s\n' "$(date -Is)"
    mkdir -p "$out/asset-attempts"
    if bash "$entry/prepare_24.sh" "$baseline" "$source" "$datasets" "$out"; then
        exit 0
    else
        mv -- "$out/assets.exit" "$out/asset-attempts/$(date +%Y%m%dT%H%M%S).exit"
        sleep "$delay"
    fi
done
printf '%s\n' 1 > "$out/assets.exit"
exit 1

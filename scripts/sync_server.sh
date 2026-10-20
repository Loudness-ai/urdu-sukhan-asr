#!/usr/bin/env bash
# Copy code, proofread clips and manifests to the GPU server, or bring trained adapters back.
#   bash scripts/sync_server.sh up   administrator@1.2.3.4   # code + data -> server:~/whisper-lite
#   bash scripts/sync_server.sh down administrator@1.2.3.4   # server:models/runs/* -> here (adapters + reports)
# rsync skips files that already arrived, so rerun it after a dropped connection.
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:?up or down}; host=${2:?user@host}
ssh_cmd="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
PATHS=(scripts wlite data/proofread data/fleurs projects/novel/clips models/runs/novel-r3/adapter)
case "$dir" in
  up)
    $ssh_cmd "$host" "mkdir -p whisper-lite $(printf 'whisper-lite/%s ' "${PATHS[@]}")"
    rsync -az -e "$ssh_cmd" requirements.txt "$host":whisper-lite/
    for p in "${PATHS[@]}"; do
      echo "$p"
      rsync -az -e "$ssh_cmd" --exclude __pycache__ --exclude '*.part' "$p/" "$host:whisper-lite/$p/"
    done
    echo uploaded ;;
  down)
    mkdir -p models/runs
    rsync -az -e "$ssh_cmd" --exclude enc_cache --exclude '*-merged' "$host:whisper-lite/models/runs/" models/runs/ ;;
  *) echo "first argument: up or down"; exit 1 ;;
esac

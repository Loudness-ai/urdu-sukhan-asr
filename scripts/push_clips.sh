#!/usr/bin/env bash
# Send the proofread clips that the training manifests use to the GPU server, skipping ones that
# already arrived whole. Retries on a dropped connection, so it survives a flaky network.
#   bash scripts/push_clips.sh administrator@1.2.3.4
set -uo pipefail
cd "$(dirname "$0")/.."
host=${1:?user@host}
ssh_opts=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ConnectTimeout=20)
needed=$(.venv/bin/python - <<'EOF'
import json, os
from pathlib import Path
d = Path("data/proofread/novel")
seen = set()
for m in d.glob("*.jsonl"):
    for l in m.read_text("utf-8").splitlines():
        if l.strip():
            p = os.path.normpath(d / json.loads(l)["audio"])
            if p not in seen:
                seen.add(p); print(p)
EOF
)
for attempt in $(seq 1 30); do
  remote=$(ssh "${ssh_opts[@]}" "$host" 'cd whisper-lite 2>/dev/null && find projects -type f -printf "%p %s\n"' 2>/dev/null) || { sleep 10; continue; }
  missing=()
  while read -r f; do
    size=$(stat -f %z "$f")
    grep -qxF "$f $size" <<<"$remote" || missing+=("$f")
  done <<<"$needed"
  echo "attempt $attempt: ${#missing[@]} of $(wc -l <<<"$needed" | tr -d ' ') clips to send"
  [ ${#missing[@]} -eq 0 ] && { echo "all clips on the server"; exit 0; }
  COPYFILE_DISABLE=1 tar czf - "${missing[@]:0:150}" | ssh "${ssh_opts[@]}" "$host" 'mkdir -p whisper-lite && tar xzf - -C whisper-lite' || sleep 10
done
echo "gave up after 30 attempts"; exit 1

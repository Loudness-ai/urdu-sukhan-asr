#!/usr/bin/env bash
# After compare_r6.sh, stress-test the old blend and round 6 in whisper.cpp on this Mac (GPU), as the app runs.
# Runs detached: nohup bash scripts/mac_stress_r6.sh > results/mac_stress_r6.log 2>&1 &
cd "$(dirname "$0")/.."
until grep -qE "COMPARE_DONE|Traceback" results/compare_r6.log || ! pgrep -f "[c]ompare_r6.sh" >/dev/null; do sleep 15; done
S=data/stress; M=models/ggml
.venv/bin/python scripts/cpp_stress.py --models $M/ggml-blend-r4r5-q8_0.bin $M/ggml-r6c2775-q8_0.bin \
  --manifests $S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl $S/phone.jsonl $S/short.jsonl $S/mixed.jsonl \
  --name mac_stress_r6 --threads 4 2>&1 | grep -vE "^\s*$"
echo MAC_STRESS_DONE

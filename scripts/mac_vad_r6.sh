#!/usr/bin/env bash
# Round 6 in whisper.cpp on this Mac with the Silero voice detector on: stress clips, then real speech
# (Common Voice new voices, both books) to make sure the detector doesn't cut words.
# Runs detached: nohup bash scripts/mac_vad_r6.sh > results/mac_vad_r6.log 2>&1 &
cd "$(dirname "$0")/.."
S=data/stress
.venv/bin/python scripts/cpp_stress.py --models models/ggml/ggml-r6c2775-q8_0.bin --vad models/vad/ggml-silero-v5.1.2.bin \
  --manifests $S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl $S/phone.jsonl $S/short.jsonl $S/mixed.jsonl \
              data/proofread/novel/test_book1.jsonl data/proofread/novel/test_book2.jsonl data/cv20k/test_unique.jsonl \
  --name mac_vad_r6 --threads 4 2>&1 | grep -vE "^\s*$"
echo MAC_VAD_DONE

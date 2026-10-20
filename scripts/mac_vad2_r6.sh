#!/usr/bin/env bash
# Voice detector that only splits at long pauses (2 s) with 300 ms padding: does it keep mixed Urdu+English whole?
cd "$(dirname "$0")/.."
S=data/stress
.venv/bin/python scripts/cpp_stress.py --models models/ggml/ggml-r6c2775-q8_0.bin --vad models/vad/ggml-silero-v5.1.2.bin \
  --vad-min-silence-ms 2000 --vad-pad-ms 300 \
  --manifests $S/mixed.jsonl $S/noise.jsonl $S/silence.jsonl $S/quiet.jsonl $S/short.jsonl \
  --name mac_vad2_r6 --threads 4 2>&1 | grep -vE "^\s*$|HF Hub"
echo MAC_VAD2_DONE

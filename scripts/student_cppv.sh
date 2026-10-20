#!/usr/bin/env bash
# App-engine (whisper.cpp CPU, beam 5, fallback, no timestamps) stress test, optionally behind the Silero
# voice detector as the live page now runs it.
#   bash scripts/student_cppv.sh <name> <threads> <vad|novad> <model.bin> [more models]
cd "$(dirname "$0")/.."
export PATH=$HOME/src/whisper.cpp/build/bin:$PATH
name=$1; threads=$2; mode=$3; shift 3
VAD=()
[ "$mode" = vad ] && VAD=(--vad models/vad/ggml-silero-v5.1.2.bin --vad-threshold 0.5 --vad-min-silence-ms 2000 --vad-pad-ms 300)
S=data/stress
.venv/bin/python scripts/cpp_stress.py --models "$@" "${VAD[@]}" \
  --manifests $S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl $S/phone.jsonl $S/short.jsonl $S/english.jsonl $S/mixed.jsonl $S/long.jsonl \
  --name $name --threads $threads 2>&1 | grep --line-buffered -vE "^\s*$"
echo CPP_DONE

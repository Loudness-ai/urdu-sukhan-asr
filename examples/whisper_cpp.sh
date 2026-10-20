#!/usr/bin/env bash
# Set up whisper.cpp with Urdu Sukhan ASR v1 and transcribe a file, with the settings we recommend.
#
#   bash examples/whisper_cpp.sh speech.wav
#   QUANT=q4_0 THREADS=8 bash examples/whisper_cpp.sh interview.mp3
#
# The first run builds whisper.cpp v1.9.5 with a small patch that adds the -nsb switch (allow an empty result, so
# silence gives no text), and downloads the model and the Silero voice detector into $DIR. Needs git, cmake, a C++
# compiler and the Hugging Face CLI (pip install huggingface_hub). Reads WAV, FLAC, MP3 and OGG.
#
# QUANT: q8_0 (default; best on Apple Silicon), q4_0 (fastest on x86 servers), q5_0.
set -euo pipefail

AUDIO=$(cd "$(dirname "${1:?usage: whisper_cpp.sh AUDIO_FILE}")" && pwd)/$(basename "$1")
QUANT=${QUANT:-q8_0}
THREADS=${THREADS:-8}
LANGUAGE=${LANGUAGE:-ur}   # ur, en, or auto (auto runs the encoder twice per window)
DIR=${DIR:-$HOME/urdu-sukhan-asr}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$DIR"
cd "$DIR"

if [ ! -x whisper.cpp/build/bin/whisper-cli ]; then
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp
  git -C whisper.cpp apply "$HERE/whisper-cpp-no-suppress-blank.patch"
  cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build whisper.cpp/build -j --config Release --target whisper-cli > /dev/null
fi

MODEL=models/ggml/ggml-urdu-sukhan-asr-v1-$QUANT.bin
[ -f "$MODEL" ] || hf download Loudness-ai/urdu-sukhan-asr-v1 "ggml/ggml-urdu-sukhan-asr-v1-$QUANT.bin" --local-dir models
VAD=whisper.cpp/models/ggml-silero-v5.1.2.bin
[ -f "$VAD" ] || bash whisper.cpp/models/download-vad-model.sh silero-v5.1.2 whisper.cpp/models > /dev/null

# -bs 5: beam search      -nt: no timestamps      -mc 0: no text carried between windows
# -nsb: silence gives no text      --vad: skip non-speech; -vsd/-vp keep short pauses inside speech
whisper.cpp/build/bin/whisper-cli -m "$MODEL" -l "$LANGUAGE" -t "$THREADS" -bs 5 -nt -mc 0 -nsb \
  --vad -vm "$VAD" -vsd 2000 -vp 300 -f "$AUDIO" 2> whisper-cli.log

#!/usr/bin/env bash
# CPU speed and accuracy test for whisper.cpp models, run the way the app runs them
# (beam 5, temperature fallback, no timestamps, language auto-detect, no carried context).
#
#   bash speed_test.sh            # every model in models/, all CPU cores
#   bash speed_test.sh 4          # 4 threads (like a phone or a small VPS)
#
# Needs whisper-cli and whisper-bench on PATH (see README.txt), or WHISPER_BIN=/path/to/whisper.cpp/build/bin.
# Speed: encoder time for one 30 s window (whisper-bench) and wall time for all test clips.
# Accuracy: python3 score.py afterwards (no installs needed).
set -euo pipefail
cd "$(dirname "$0")"
T=${1:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
BIN=${WHISPER_BIN:+$WHISPER_BIN/}
CLI=${BIN}whisper-cli
BENCH=${BIN}whisper-bench
command -v "$CLI" >/dev/null || { echo "whisper-cli not found: build whisper.cpp first (README.txt)"; exit 1; }
AUDIO=$(awk -F'\t' 'NR > 1 {s += $2} END {printf "%.1f", s}' clips/refs.tsv)
N=$(($(wc -l < clips/refs.tsv) - 1))
FILES=$(awk -F'\t' 'NR > 1 {printf "-f clips/%s ", $1}' clips/refs.tsv)
CPU=$( (lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}') || true)
echo "CPU: ${CPU:-$(sysctl -n machdep.cpu.brand_string 2>/dev/null)}"
echo "threads: $T   test clips: $N ($AUDIO s of audio)"
echo
printf "%-28s %14s %12s %14s\n" model "encoder 30 s" "all clips" "x real time"
for m in models/*.bin; do
  name=$(basename "$m" .bin)
  enc=$("$BENCH" -m "$m" -t "$T" 2>&1 | awk '/encode time/ {printf "%.2f s", $(NF-3) / 1000}')
  rm -f clips/*.txt
  start=$(date +%s.%N)
  # shellcheck disable=SC2086
  "$CLI" -m "$m" -l auto -bs 5 -nt -mc 0 -t "$T" -np -otxt $FILES > /dev/null 2>&1
  end=$(date +%s.%N)
  mkdir -p "results/$name"
  mv clips/*.txt "results/$name/"
  wall=$(awk -v a="$start" -v b="$end" 'BEGIN {printf "%.1f", b - a}')
  rt=$(awk -v w="$wall" -v a="$AUDIO" 'BEGIN {printf "%.2f", a / w}')
  printf "%-28s %14s %10s s %13sx\n" "$name" "$enc" "$wall" "$rt"
done
echo
echo "'x real time' above 1 = faster than the audio plays. Now run: python3 score.py"

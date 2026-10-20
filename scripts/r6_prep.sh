#!/usr/bin/env bash
# Round-6 preparation on the GPU server, started detached so a dropped SSH connection can't stop it:
#   setsid nohup bash scripts/r6_prep.sh > r6_prep.log 2>&1 < /dev/null &
# 1) finish relabeling UAT with round 4 (resumes where it stopped), on the GPU
# 2) at the same time, download MUSAN (noise, music and speech-babble, CC BY 4.0) for no-speech examples
cd "$(dirname "$0")/.."
(
  .venv/bin/python scripts/pseudo_label.py data/uat/all.jsonl --out data/uat/labeled_r4.jsonl \
    --base models/hf/whisper-large-v3-turbo --adapter models/runs/novel-r4/adapter --batch 32 > relabel.log 2>&1
  echo "relabel exit $?"
) &
(
  mkdir -p data/musan
  for i in 1 2 3 4 5; do
    curl -sSL -C - -o data/musan/musan.tar.gz https://openslr.elda.org/resources/17/musan.tar.gz && break
    sleep 30
  done
  tar xzf data/musan/musan.tar.gz -C data/musan && rm -f data/musan/musan.tar.gz && echo "musan ok"
) > musan.log 2>&1 &
wait
echo R6_PREP_DONE

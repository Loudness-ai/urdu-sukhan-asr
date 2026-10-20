#!/usr/bin/env bash
# Inspect one Urdu file from each candidate dataset and score round 4 on 40 random clips of each.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/check_hf_sample.py ai4bharat/IndicVoices urdu/valid-00000-of-00001.parquet --name iv_urdu
.venv/bin/python scripts/check_hf_sample.py humairawan/UrduSpeech Urdu/test-00000-of-00007.parquet --name us_urdu
.venv/bin/python scripts/eval_hf.py --base models/hf/whisper-large-v3-turbo --models r4=models/runs/novel-r4/adapter \
  --manifests data/check/iv_urdu/sample.jsonl data/check/us_urdu/sample.jsonl --name check_sets 2>&1 | grep -E "WER|^\|"
echo CHECK_DONE

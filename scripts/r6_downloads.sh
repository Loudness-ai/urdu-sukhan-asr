#!/usr/bin/env bash
# Round-6 data downloads on the GPU server (run detached). Test files of each set are left out.
cd "$(dirname "$0")/.."
rm -rf data/iv_smoke
.venv/bin/python scripts/get_hf_audio.py ai4bharat/IndicVoices --config urdu --exclude valid --shuffle \
  --per-shard 400 --max-clips 25000 --min-seconds 0.5 --name indicvoices_ur > dl_indicvoices.log 2>&1
.venv/bin/python scripts/get_hf_audio.py humairawan/UrduSpeech --config Urdu --exclude test --shuffle \
  --per-shard 100 --max-clips 5000 --min-seconds 0.5 --name urduspeech > dl_urduspeech.log 2>&1
.venv/bin/python scripts/get_fleurs.py en_us --splits train test > dl_fleurs_en.log 2>&1
echo DOWNLOADS_DONE

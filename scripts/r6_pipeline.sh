#!/usr/bin/env bash
# Round-6 preparation, hands-off, on the GPU server (run detached). Stops after a short test run so a
# person can check it before the long training starts.
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }
log "waiting for MUSAN, IndicVoices and UrduSpeech"
until grep -q "musan ok" musan.log 2>/dev/null; do sleep 30; done
while pgrep -f "[g]et_hf_audio.py" >/dev/null; do sleep 30; done
log "downloads done: $(wc -l < data/indicvoices_ur/all.jsonl) IndicVoices, $(wc -l < data/urduspeech/all.jsonl) UrduSpeech"
.venv/bin/python scripts/make_r6_extras.py stress || { log "STRESS BUILD FAILED"; exit 1; }
.venv/bin/python scripts/make_r6_extras.py train || { log "EXTRAS BUILD FAILED"; exit 1; }
.venv/bin/python scripts/make_r6_train.py || { log "CLEANING FAILED"; exit 1; }
S=data/stress
log "scoring current models on the stress test"
.venv/bin/python scripts/eval_hf.py --base models/hf/whisper-large-v3-turbo \
  --models stock r4=models/runs/novel-r4/adapter r5=models/runs/novel-r5/model blend=models/runs/soup-r4r5-50/model \
  --manifests $S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl $S/phone.jsonl $S/short.jsonl $S/english.jsonl $S/mixed.jsonl $S/long.jsonl \
  --name stress_baseline 2>&1 | grep -E "WER|Traceback"
log "15-step test run of the round-6 recipe"
R=data/r6
rm -rf models/runs/r6-smoke
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train.py --base models/hf/whisper-large-v3-turbo \
  --train $R/book1.jsonl:2 $R/book2.jsonl:4 $R/nzg73.jsonl:~300 $R/cv.jsonl:~300 $R/uat.jsonl:~300 $R/indicvoices.jsonl:~300 \
          $R/urduspeech.jsonl:~100 $R/fleurs_en.jsonl:~100 $R/codeswitch.jsonl:~50 $R/nospeech.jsonl:~200 \
  --out models/runs/r6-smoke --full --spec-augment --max-steps 15 --save-every 10 --batch 8 --accum 2 --grad-ckpt \
  --augment 0.5 --timestamp-frac 0.3 --workers 10 --lr 5e-6 --skip-baseline \
  --noise-dir data/musan/musan/noise data/musan/musan/music 2>&1 | grep -vE "Warn|warn"
ls models/runs/r6-smoke
log "R6_PIPELINE_DONE"

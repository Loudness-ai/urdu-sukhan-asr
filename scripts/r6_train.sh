#!/usr/bin/env bash
# Stress-test baseline for today's models, then round-6 training (run detached on the GPU server).
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }
S=data/stress; R=data/r6
log "stress baseline"
.venv/bin/python scripts/eval_hf.py --base models/hf/whisper-large-v3-turbo \
  --models stock r4=models/runs/novel-r4/adapter r5=models/runs/novel-r5/model blend=models/runs/soup-r4r5-50/model \
  --manifests $S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl $S/phone.jsonl $S/short.jsonl $S/english.jsonl $S/mixed.jsonl $S/long.jsonl \
  --name stress_baseline 2>&1 | grep -E "WER|Traceback"
log "round 6 training"
mkdir -p models/runs/novel-r6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train.py --base models/hf/whisper-large-v3-turbo \
  --train $R/book1.jsonl:2 $R/book2.jsonl:4 $R/nzg73.jsonl:~3000 $R/fleurs_ur.jsonl:1 $R/fleurs_ur_dev.jsonl:1 \
          $R/cv.jsonl:~12000 $R/uat.jsonl:~10000 $R/indicvoices.jsonl:~20000 $R/urduspeech.jsonl:1 \
          $R/fleurs_en.jsonl:~2000 $R/codeswitch.jsonl:1 $R/nospeech.jsonl:1 \
  --eval data/proofread/novel/test_book2.jsonl data/tests/indicvoices.jsonl:100 \
  --out models/runs/novel-r6 --full --spec-augment --epochs 1 --save-every 925 --batch 8 --accum 2 --grad-ckpt \
  --augment 0.5 --timestamp-frac 0.3 --workers 10 --lr 5e-6 --skip-baseline \
  --noise-dir data/musan/musan/noise data/musan/musan/music > models/runs/novel-r6/train.log 2>&1
log "R6_TRAIN_DONE"

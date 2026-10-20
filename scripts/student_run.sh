#!/usr/bin/env bash
# Student distillation on the GPU server (run detached). Stages:
#   bash scripts/student_run.sh smoke              # 10 steps on the full data list
#   bash scripts/student_run.sh pilot              # e16d4 and e8d4, 600 steps each, one after the other
#   bash scripts/student_run.sh main <init> <steps> <out>   # the long run, from a pilot checkpoint
cd "$(dirname "$0")/.."
R=data/r6
TRAIN=($R/book1.jsonl:3 $R/book2.jsonl:10 $R/nzg73.jsonl:~4000 $R/fleurs_ur.jsonl $R/fleurs_ur_dev.jsonl $R/cv.jsonl
       $R/uat.jsonl:~12000 $R/indicvoices.jsonl $R/urduspeech.jsonl $R/fleurs_en.jsonl $R/codeswitch.jsonl $R/nospeech.jsonl)
COMMON=(--teacher models/runs/novel-r6/ckpt-2775 --train "${TRAIN[@]}" --batch 8 --accum 2 --grad-ckpt
        --augment 0.5 --spec-augment --noise-dir data/musan/musan/noise data/musan/musan/music --workers 10 --dev-max 60)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { .venv/bin/python scripts/train_student.py "${COMMON[@]}" "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Writing model|max_new_tokens|clean_up_tokenization|^\s*$"; }
log() { echo "[$(date +%H:%M)] $*"; }
case "$1" in
  smoke)
    rm -rf models/runs/smoke-full
    run --student models/students/init-e8d4 --out models/runs/smoke-full --max-steps 10 --save-every 0 \
        --quick-sets data/stress/silence.jsonl:4 --dev-max 2
    log SMOKE_DONE ;;
  pilot)
    for n in e16d4 e8d4; do
      log "pilot $n"
      mkdir -p models/runs/pilot-$n
      run --student models/students/init-$n --out models/runs/pilot-$n --max-steps 600 --save-every 300 > models/runs/pilot-$n/train.log
    done
    log PILOT_DONE ;;
  main)
    init=${2:?init folder}; steps=${3:?steps}; out=${4:?out folder}
    mkdir -p "$out"
    log "main from $init, $steps steps -> $out"
    run --student "$init" --out "$out" --max-steps "$steps" --save-every 300 > "$out/train.log"
    log MAIN_DONE ;;
  *) echo "smoke | pilot | main <init> <steps> <out>"; exit 1 ;;
esac

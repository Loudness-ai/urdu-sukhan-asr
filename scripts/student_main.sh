#!/usr/bin/env bash
# Student distillation, main run (GPU server, run detached), with clips levelled before augmentation and
# code-switch windows built on the fly instead of data/r6/codeswitch.jsonl.
#   bash scripts/student_main.sh smoke <init> [extra args]    # 10 steps on the full data list
#   bash scripts/student_main.sh run <init> <steps> <out>     # e.g. from a pilot checkpoint
cd "$(dirname "$0")/.."
R=data/r6
TRAIN=($R/book1.jsonl:3 $R/book2.jsonl:10 $R/nzg73.jsonl:~4000 $R/fleurs_ur.jsonl $R/fleurs_ur_dev.jsonl $R/cv.jsonl
       $R/uat.jsonl:~12000 $R/indicvoices.jsonl $R/urduspeech.jsonl $R/fleurs_en.jsonl $R/nospeech.jsonl)
COMMON=(--teacher models/runs/novel-r6/ckpt-2775 --train "${TRAIN[@]}" --codeswitch 0.02 --batch 8 --accum 2 --grad-ckpt
        --augment 0.5 --spec-augment --noise-dir data/musan/musan/noise data/musan/musan/music --workers 10 --dev-max 60)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { .venv/bin/python scripts/train_student.py "${COMMON[@]}" "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Writing model|max_new_tokens|clean_up_tokenization|^\s*$"; }
log() { echo "[$(date +%H:%M)] $*"; }
case "$1" in
  smoke)
    rm -rf models/runs/smoke-main
    run --student "${2:?init}" --out models/runs/smoke-main --max-steps 10 --save-every 0 \
        --quick-sets data/stress/silence.jsonl:4 --dev-max 2 "${@:3}"
    log SMOKE_DONE ;;
  run)
    init=${2:?init}; steps=${3:?steps}; out=${4:?out}
    mkdir -p "$out"
    log "main from $init, $steps steps -> $out"
    run --student "$init" --out "$out" --max-steps "$steps" --save-every 300 "${@:5}" > "$out/train.log"
    log MAIN_DONE ;;
  *) echo "smoke <init> | run <init> <steps> <out> [extra args]"; exit 1 ;;
esac

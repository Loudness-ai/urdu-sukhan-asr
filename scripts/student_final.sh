#!/usr/bin/env bash
# Score the chosen student next to the teacher, then export it (GPU server, run detached).
#   bash scripts/student_final.sh hf <ckpt> <label>        # GPU: 7 tests + 9 stress, raw and level-normalised
#   bash scripts/student_final.sh export <ckpt> <label>    # ggml q8_0 + q5_0, Core ML encoder (CPU)
#   bash scripts/student_final.sh cpp <label> <threads>    # whisper.cpp stress, without and with the voice detector
cd "$(dirname "$0")/.."
T=models/runs/novel-r6/ckpt-2775
S=data/stress
TESTS=(data/proofread/novel/test_book1.jsonl data/proofread/novel/test_book2.jsonl data/fleurs/ur_pk/test.jsonl data/nzg73/test.jsonl
       data/cv20k/test_unique.jsonl data/tests/indicvoices.jsonl data/tests/urduspeech.jsonl)
STRESS=($S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl $S/phone.jsonl $S/short.jsonl $S/english.jsonl $S/mixed.jsonl $S/long.jsonl)
export PATH=$HOME/src/whisper.cpp/build/bin:$PATH
quiet() { grep --line-buffered -vE "Loading weights|max_new_tokens|clean_up_tokenization|attention mask|generation_config|^\s*$"; }
log() { echo "[$(date +%H:%M)] $*"; }
case "$1" in
  hf)
    ck=${2:?ckpt}; label=${3:?label}
    log "HF raw"
    .venv/bin/python scripts/eval_hf.py --base $T --models teacher=$T $label=$ck --manifests "${TESTS[@]}" "${STRESS[@]}" \
      --name student_vs_teacher 2>&1 | quiet
    log "HF level-normalised (as an app that evens out volume before the model)"
    .venv/bin/python scripts/eval_hf.py --base $T --models teacher=$T $label=$ck --normalize \
      --manifests data/fleurs/ur_pk/test.jsonl data/cv20k/test_unique.jsonl $S/quiet.jsonl $S/english.jsonl $S/mixed.jsonl \
      --name student_vs_teacher_norm 2>&1 | quiet
    log "HF Book 2 with the old blend"
    .venv/bin/python scripts/eval_hf.py --base $T --models blend=models/runs/soup-r4r5-50/model \
      --manifests data/proofread/novel/test_book2.jsonl --name student_book2_blend 2>&1 | quiet
    .venv/bin/python scripts/name_check.py teacher=results/student_vs_teacher/teacher__novel_test_book2.jsonl \
      $label=results/student_vs_teacher/${label}__novel_test_book2.jsonl blend=results/student_book2_blend/blend__novel_test_book2.jsonl
    log HF_DONE ;;
  export)
    ck=${2:?ckpt}; label=${3:?label}
    .venv/bin/python scripts/export_ggml.py --base models/hf/whisper-large-v3-turbo --model $ck --name $label --quant q8_0 q5_0
    ~/coreml-venv/bin/python scripts/export_coreml.py --model $ck --name $label 2>&1 | grep -E "done|check|Error|Traceback"
    log EXPORT_DONE ;;
  cpp)
    label=${2:?label}; threads=${3:-12}
    bash scripts/student_cppv.sh cpp_student_${label} $threads novad models/ggml/ggml-$label-q8_0.bin
    bash scripts/student_cppv.sh cpp_student_${label}_vad $threads vad models/ggml/ggml-$label-q8_0.bin
    .venv/bin/python scripts/book_prompt_check.py --models models/ggml/ggml-$label-q8_0.bin --name book2_prompt_$label --threads $threads
    log CPP_ALL_DONE ;;
  *) echo "hf <ckpt> <label> | export <ckpt> <label> | cpp <label> [threads]"; exit 1 ;;
esac

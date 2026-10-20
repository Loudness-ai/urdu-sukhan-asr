#!/usr/bin/env bash
# While round 6 trains on the GPU, score each new checkpoint on the stress test on the CPU (the GPU has
# no room for a second model). Low priority and 8 threads so training's data loading isn't slowed.
cd "$(dirname "$0")/.."
S=data/stress
for step in 925 1850 2775; do
  ck=models/runs/novel-r6/ckpt-$step
  until [ -f $ck/model.safetensors ] && [ -f $ck/tokenizer_config.json ]; do
    pgrep -f "[t]rain.py" >/dev/null || exit 0
    sleep 60
  done
  sleep 30
  echo "[$(date +%H:%M)] scoring ckpt-$step"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice -n 19 .venv/bin/python scripts/eval_hf.py --base models/hf/whisper-large-v3-turbo \
    --models c$step=$ck --manifests $S/noise.jsonl $S/music.jsonl $S/silence.jsonl $S/quiet.jsonl:20 $S/short.jsonl $S/mixed.jsonl \
    --name ckpt$step --batch 8 2>&1 | grep -E "WER|Traceback"
done
echo CKPT_WATCH_DONE

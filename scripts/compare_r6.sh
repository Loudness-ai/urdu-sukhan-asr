#!/usr/bin/env bash
# Score round 6 (checkpoint 2775) in whisper.cpp on this Mac (GPU) on unheard Common Voice speakers and both
# books, next to the earlier models' saved results, then count loops with the current rule.
# Runs detached: nohup bash scripts/compare_r6.sh > results/compare_r6.log 2>&1 &
cd "$(dirname "$0")/.."
M=models/ggml/ggml-r6c2775-q8_0.bin
.venv/bin/python scripts/evaluate.py data/cv20k/test_unique.jsonl --no-timestamps --name mac_cv --models $M 2>&1 | grep -E "^\||Traceback"
for book in 1 2; do
  .venv/bin/python scripts/evaluate.py data/proofread/novel/test_book$book.jsonl --no-timestamps --name mac_book${book}_r5 --models $M 2>&1 | grep -E "^\||Traceback"
done
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from eval_hf import runaway
for test in ("mac_cv", "mac_book1_r5", "mac_book2_r5"):
    for f in ("blend-r4r5-q8_0", "novel-r5-q8_0", "r6c2775-q8_0"):
        try:
            rows = [json.loads(l) for l in open(f"results/{test}/{f}.jsonl")]
        except FileNotFoundError:
            continue
        print(f"{test:14s} {f:18s} loops {sum(runaway(r['hyp'], r['ref']) for r in rows)}/{len(rows)}")
PY
echo COMPARE_DONE

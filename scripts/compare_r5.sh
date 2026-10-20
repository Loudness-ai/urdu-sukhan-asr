#!/usr/bin/env bash
# Score round 5, the round 4+5 blend, round 4 and stock in whisper.cpp on this Mac (GPU), on unheard
# Common Voice speakers and both books, then count runaway (looping) outputs.
# Runs detached: bash scripts/compare_r5.sh > results/compare_r5.log 2>&1 &
cd "$(dirname "$0")/.."
M=models/ggml
.venv/bin/python scripts/evaluate.py data/cv20k/test_unique.jsonl --no-timestamps --name mac_cv \
  --models $M/ggml-novel-r5-q8_0.bin $M/ggml-blend-r4r5-q8_0.bin $M/ggml-novel-r4-q8_0.bin large-v3-turbo-q8_0 2>&1 | grep -E "^\||Traceback"
for book in 1 2; do
  .venv/bin/python scripts/evaluate.py data/proofread/novel/test_book$book.jsonl --no-timestamps --name mac_book${book}_r5 \
    --models $M/ggml-novel-r5-q8_0.bin $M/ggml-blend-r4r5-q8_0.bin 2>&1 | grep -E "^\||Traceback"
done
.venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, ".")
from wlite.text import word_char_errors
loop = lambda r: len(r["hyp"].split()) > 2 * max(3, len(r["ref"].split()))
files = {"stock": "large-v3-turbo-q8_0", "r4": "novel-r4-q8_0", "r5": "novel-r5-q8_0", "blend": "blend-r4r5-q8_0"}
d = {n: {r["id"]: r for r in map(json.loads, open(f"results/mac_cv/{f}.jsonl"))} for n, f in files.items()}
ok = [i for i in d["r4"] if not any(loop(m[i]) for m in d.values())]
for n, m in d.items():
    w, c = word_char_errors([m[i]["ref"] for i in ok], [m[i]["hyp"] for i in ok], "ur")
    print(f"{n:6s} loops {sum(loop(r) for r in m.values()):2d}/{len(m)} | on the {len(ok)} loop-free clips: WER {w:.1f}%  CER {c:.1f}%")
EOF
echo COMPARE_DONE

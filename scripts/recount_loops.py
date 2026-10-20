"""Recount loops in saved eval_hf outputs with the current loop rule, without running any model again.

  python scripts/recount_loops.py results/baseline_r6 results/r6_tests
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hf import runaway  # noqa: E402


def main():
    table = defaultdict(dict)
    for folder in map(Path, sys.argv[1:]):
        for f in sorted(folder.glob("*__*.jsonl")):
            model, test = f.stem.split("__", 1)
            rows = [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()]
            speech = [r for r in rows if r["ref"].strip()]
            table[test][model] = f"{sum(runaway(r['hyp'], r['ref']) for r in speech)}/{len(speech)}" if speech else "-"
    models = sorted({m for t in table.values() for m in t}, key=lambda m: (m.startswith("c"), m))
    print("| Test | " + " | ".join(models) + " |")
    print("|---|" + "---:|" * len(models))
    for test, row in table.items():
        print(f"| {test} | " + " | ".join(row.get(m, "") for m in models) + " |")


if __name__ == "__main__":
    main()

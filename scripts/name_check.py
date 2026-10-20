"""How many of a book's character names come out spelled right, per model.

  python scripts/name_check.py results/x/teacher__novel_test_book2.jsonl results/x/student__novel_test_book2.jsonl

Reads result files with "ref" and "hyp" (from eval_hf.py or cpp_stress.py). For every name occurrence in a
reference, it counts as right when the hypothesis of that clip has the name as often (word-bounded, after
the same normalisation as WER scoring). Prints a table: right / said, per name and in total.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.text import normalize_for_eval  # noqa: E402

BOOK2 = ["نور الہدیٰ", "اظہر فاروقی", "عمیر", "عزیر", "تانیا", "فائزہ", "شایان", "خدیجہ", "سعدیہ راجپوت", "ملیحہ", "طارق"]


def count(name, text):
    return len(re.findall(rf"(?<!\S){re.escape(name)}(?!\S)", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="label=path or path (label = file name before '__')")
    ap.add_argument("--names", nargs="*", default=BOOK2)
    args = ap.parse_args()

    names = [normalize_for_eval(n, "ur") for n in args.names]
    table = {}
    for spec in args.results:
        label, _, path = spec.rpartition("=") if "=" in spec else (Path(spec).name.split("__")[0], "", spec)
        rows = [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]
        per = {}
        for r in rows:
            ref, hyp = normalize_for_eval(r["ref"], "ur"), normalize_for_eval(r["hyp"], "ur")
            for n in names:
                said = count(n, ref)
                if said:
                    got = per.setdefault(n, [0, 0])
                    got[0] += min(said, count(n, hyp))
                    got[1] += said
        table[label] = per
    used = [n for n in names if any(n in per for per in table.values())]
    lines = ["| model | " + " | ".join(used) + " | all names |", "|---" * (len(used) + 2) + "|"]
    for label, per in table.items():
        right, said = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        lines.append(f"| {label} | " + " | ".join(f"{per[n][0]}/{per[n][1]}" if n in per else "–" for n in used)
                     + f" | {right}/{said} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

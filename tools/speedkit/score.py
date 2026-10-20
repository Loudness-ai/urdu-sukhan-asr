"""Accuracy of each model's transcripts from speed_test.sh, per test set. Plain Python 3, nothing to install.

  python3 score.py

Words wrong (WER) and letters wrong (CER) after the same normalisation the project's scoring uses (punctuation,
diacritics, digit forms and hamza variants ignored). For no-speech clips: how many got any text, and how many got
two or more words ("invented text").
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import normalize_for_eval  # noqa: E402

HERE = Path(__file__).resolve().parent


def edits(a, b):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    rows = list(csv.DictReader(open(HERE / "clips" / "refs.tsv", encoding="utf-8"), delimiter="\t"))
    models = sorted(p.name for p in (HERE / "results").iterdir() if p.is_dir())
    sets = sorted({r["set"] for r in rows}, key=lambda s: (s.startswith("no-speech"), s))
    print(f"{'test set':24s}" + "".join(f"{m:>26s}" for m in models))
    for s in sets:
        part = [r for r in rows if r["set"] == s]
        cells = []
        for m in models:
            hyp = {r["file"]: (HERE / "results" / m / (r["file"] + ".txt")) for r in part}
            hyp = {k: (p.read_text("utf-8", errors="replace").strip() if p.exists() else "") for k, p in hyp.items()}
            if not part[0]["text"]:
                anyw = sum(len(hyp[r["file"]].split()) >= 1 for r in part)
                two = sum(len(hyp[r["file"]].split()) >= 2 for r in part)
                cells.append(f"any {anyw}/{len(part)}, 2+ {two}/{len(part)}")
                continue
            we = wn = ce = cn = 0
            for r in part:
                lang = r["language"]
                ref, h = normalize_for_eval(r["text"], lang), normalize_for_eval(hyp[r["file"]], lang)
                we += edits(ref.split(), h.split())
                wn += len(ref.split())
                ce += edits(ref.replace(" ", ""), h.replace(" ", ""))
                cn += len(ref.replace(" ", ""))
            cells.append(f"{100 * we / wn:5.1f}% words, {100 * ce / cn:4.1f}% letters")
        print(f"{s + f' ({len(part)})':24s}" + "".join(f"{c:>26s}" for c in cells))
    print("\nLower is better. Each model's raw transcripts are in results/<model>/<clip>.txt; references in clips/refs.tsv.")


if __name__ == "__main__":
    main()

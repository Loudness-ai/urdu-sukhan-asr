"""Clean every round-6 training source and write one manifest per source into data/r6/.

  python scripts/make_r6_train.py

A speech clip is dropped when its audio can't hold its text (too many letters per second, the
cut-off Common Voice clips that taught round 5 to invent words), when it is empty or outside
0.5-30 s, when an Urdu clip's text is mostly not in Urdu script, or when UAT's machine label was
not confident. No-speech clips (empty text) are kept as they are. Audio paths become absolute.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
ARABIC = re.compile(r"[؀-ۿ]")
LETTER = re.compile(r"[^\W\d_]")

SOURCES = {  # name: (manifest, extra rule)
    "book1": (D / "proofread/novel/train_r4_book1.jsonl", None),
    "book2": (D / "proofread/novel/train_r4_book2.jsonl", None),
    "nzg73": (D / "nzg73/train.jsonl", None),
    "fleurs_ur": (D / "fleurs/ur_pk/train.jsonl", None),
    "fleurs_ur_dev": (D / "fleurs/ur_pk/dev.jsonl", None),
    "cv": (D / "cv20k/train.jsonl", None),
    "uat": (D / "uat/labeled_r4.jsonl", lambda r: r.get("conf", 0) >= 0.85),
    "indicvoices": (D / "indicvoices_ur/all.jsonl", None),
    "urduspeech": (D / "urduspeech/all.jsonl", None),
    "fleurs_en": (D / "fleurs/en_us/train.jsonl", None),
    "codeswitch": (D / "codeswitch/train.jsonl", None),
    "nospeech": (D / "nospeech/train.jsonl", None),
}


def why_drop(r):
    text, dur = (r.get("text") or "").strip(), float(r.get("duration") or 0)
    if not 0.5 <= dur <= 30:
        return "length"
    if not text:
        return "empty"
    letters = LETTER.findall(text)
    per_second = len(letters) / dur
    if per_second > (25 if dur < 1.5 else 22) or (per_second < 1.5 and dur > 4):
        return "audio/text mismatch"
    if r.get("language", "ur") == "ur" and letters and len(ARABIC.findall(text)) / len(letters) < 0.6:
        return "not Urdu script"
    return None


def main():
    out = D / "r6"
    out.mkdir(exist_ok=True)
    total = 0
    for name, (path, rule) in SOURCES.items():
        if not path.exists():
            print(f"{name:14s} MISSING {path}")
            continue
        kept, dropped = [], {}
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["audio"] = str((path.parent / r["audio"]).resolve())
            reason = None if name == "nospeech" else why_drop(r)
            if reason == "not Urdu script" and name == "codeswitch":  # Urdu + English on purpose
                reason = None
            if reason is None and rule and not rule(r):
                reason = "unsure label"
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                kept.append(r)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), "utf-8")
        hours = sum(float(r.get("duration") or 0) for r in kept) / 3600
        total += len(kept)
        print(f"{name:14s} kept {len(kept):6d} ({hours:5.1f} h)  dropped {dropped}")
    print("total clips available:", total)


if __name__ == "__main__":
    main()

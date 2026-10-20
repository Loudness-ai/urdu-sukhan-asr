"""Turn proofread clips into train/test manifests for fine-tuning.

  python scripts/export_dataset.py batch1 [batch2 ...]

Only clips marked "done" are used. The test set is held out by recording, not
by clip: clips from one recording share a speaker and a microphone, so letting
them land on both sides would make the model look better than it is.
With fewer than 5 recordings, the last ~10% of each recording is held out instead.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.project import Project  # noqa: E402
from wlite.text import clean_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def assign_splits(items, test_frac):
    sources = sorted({it["source"] for it in items})
    split = {}
    if len(sources) >= 5:
        # Stable choice: a recording stays on the same side as more audio gets added.
        ranked = sorted(sources, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
        n_test = max(1, round(len(sources) * test_frac))
        test_sources = set(ranked[:n_test])
        for it in items:
            split[it["id"]] = "test" if it["source"] in test_sources else "train"
    else:
        for src in sources:
            clips = sorted((it for it in items if it["source"] == src), key=lambda it: it["start"])
            n_test = max(1, round(len(clips) * test_frac)) if len(clips) >= 5 else 0
            for k, it in enumerate(clips):
                split[it["id"]] = "test" if k >= len(clips) - n_test else "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--out", default=None, help="output folder (default data/proofread/<first project>)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "data" / "proofread" / args.projects[0]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.projects:
        proj = Project(name)
        for it in proj.items:
            if it["status"] == "done" and (it.get("text") or "").strip():
                rows.append((proj, it))
    split = assign_splits([it for _, it in rows], args.test_frac)

    counts = {"train": [0, 0.0], "test": [0, 0.0]}
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in counts}
    for proj, it in rows:
        s = split[it["id"]]
        rec = {
            "id": f"{proj.name}-{it['id']}",
            "audio": os.path.relpath(proj.clip_dir / it["clip"], out),  # relative to the manifest's folder
            "text": clean_label(it["text"], it["language"]),
            "language": it["language"],
            "duration": it["duration"],
            "group": it["source"],
        }
        words = [w for w in it.get("words") or [] if w.get("end", 0) > w.get("start", 0)]
        if words:  # where speech starts and ends in the clip, from the draft's word timings
            start, end = words[0]["start"] / 1000, min(words[-1]["end"] / 1000, it["duration"])
            if end - start > 0.5:
                rec["speech_start"], rec["speech_end"] = round(start, 2), round(end, 2)
        files[s].write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[s][0] += 1
        counts[s][1] += it["duration"]
    for f in files.values():
        f.close()
    for s, (n, d) in counts.items():
        print(f"{s:5s}: {n} clips, {d / 60:.1f} min -> {out / (s + '.jsonl')}")
    if counts["test"][0] == 0:
        print("note: no test clips yet; proofread more before trusting any score.")


if __name__ == "__main__":
    main()

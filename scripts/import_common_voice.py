"""Turn a Mozilla Common Voice style folder (a .tsv with client_id/path/sentence + audio files) into
train/test manifests. Test speakers never appear in training, so the score shows how the model
does on voices it has not heard.

  python scripts/import_common_voice.py data/cv20k --tsv final_main_dataset.tsv --audio-dir limited_wav_files/limited_wav_files

Keeps clips that volunteers approved (more up votes than down votes) and that have audio.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.text import clean_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tsv", default="validated.tsv")
    ap.add_argument("--audio-dir", default="clips")
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--test-speakers", type=float, default=0.05, help="share of speakers held out for testing")
    args = ap.parse_args()

    folder = Path(args.folder)
    audio_dir = folder / args.audio_dir
    files = {p.stem: p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")}
    out = {"train": [], "test": []}
    skipped = {"votes": 0, "no audio": 0, "empty": 0, "bad audio": 0}
    with open(folder / args.tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if int(row.get("up_votes") or 0) <= int(row.get("down_votes") or 0):
                skipped["votes"] += 1
                continue
            path = files.get(Path(row["path"]).stem)
            if path is None:
                skipped["no audio"] += 1
                continue
            text = clean_label(row["sentence"], args.lang)
            if not text:
                skipped["empty"] += 1
                continue
            try:
                dur = sf.info(path).duration
            except Exception:
                skipped["bad audio"] += 1
                continue
            # Same speaker -> same split, decided by a hash so reruns agree.
            h = int(hashlib.sha1(row["client_id"].encode()).hexdigest(), 16) % 10000
            split = "test" if h < args.test_speakers * 10000 else "train"
            out[split].append({"id": f"cv-{path.stem}", "audio": str(path.relative_to(folder)), "text": text,
                               "language": args.lang, "duration": round(dur, 2), "group": "common_voice",
                               "speaker": row["client_id"][:12], "gender": row.get("gender") or ""})
    for split, rows in out.items():
        with open(folder / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        speakers = len({r["speaker"] for r in rows})
        print(f"{split}: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.1f} h, {speakers} speakers")
    print("skipped:", skipped)


if __name__ == "__main__":
    main()

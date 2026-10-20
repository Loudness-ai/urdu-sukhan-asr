"""Look inside one parquet file of a Hugging Face speech dataset before using it: which columns it has,
which script the text is in (Urdu vs Hindi), clip lengths, speakers, and 40 random clips written out
with a manifest, so a model can be scored on them.

  python scripts/check_hf_sample.py ai4bharat/IndicVoices urdu/valid-00000-of-00001.parquet --name iv_urdu
"""

import argparse
import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.text import clean_label  # noqa: E402

ARABIC = re.compile(r"[؀-ۿ]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("file")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    local = hf_hub_download(args.repo, args.file, repo_type="dataset")
    table = pq.read_table(local)
    print("columns:", {f.name: str(f.type)[:40] for f in table.schema})
    rows = table.to_pylist()
    audio_col = next(f.name for f in table.schema if str(f.type).startswith("struct") and "bytes" in str(f.type))
    text_col = next((c for c in ("text", "transcription", "transcript", "sentence", "normalized") if c in table.column_names), None)
    texts = [r.get(text_col) or "" for r in rows] if text_col else []
    joined = " ".join(texts)
    a, d, l = len(ARABIC.findall(joined)), len(DEVANAGARI.findall(joined)), len(LATIN.findall(joined))
    total = max(a + d + l, 1)
    print(f"rows: {len(rows)} | text column: {text_col} | letters: Urdu/Arabic {100 * a / total:.0f}%  Hindi/Devanagari {100 * d / total:.0f}%  Latin {100 * l / total:.0f}%")
    for c in ("speaker_id", "speaker", "gender", "scenario", "task_name", "style", "emotion", "district", "state"):
        if c in table.column_names:
            vals = Counter(str(r.get(c)) for r in rows)
            print(f"{c}: {len(vals)} distinct, top {vals.most_common(4)}")

    out = ROOT / "data" / "check" / args.name
    (out / "clips").mkdir(parents=True, exist_ok=True)
    random.seed(0)
    picked = random.sample(range(len(rows)), min(args.n, len(rows)))
    durs = []
    with open(out / "sample.jsonl", "w", encoding="utf-8") as f:
        for k, i in enumerate(picked):
            wav, sr = sf.read(io.BytesIO(rows[i][audio_col]["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = soxr.resample(wav, sr, 16000) if sr != 16000 else wav
            dur = len(wav) / 16000
            durs.append(dur)
            sf.write(out / "clips" / f"{k}.flac", np.clip(wav[: 30 * 16000], -1, 1), 16000)
            f.write(json.dumps({"id": f"{args.name}-{k}", "audio": f"clips/{k}.flac", "text": clean_label(texts[i], "ur") if texts else "",
                                "raw_text": texts[i] if texts else "", "language": "ur", "duration": round(dur, 2)}, ensure_ascii=False) + "\n")
    print(f"sample clip lengths: median {np.median(durs):.1f}s, max {max(durs):.1f}s, over 30s: {sum(x > 30 for x in durs)}")
    for i in picked[:5]:
        print("  TEXT:", (texts[i] if texts else "")[:120])


if __name__ == "__main__":
    main()

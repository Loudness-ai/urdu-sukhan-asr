"""Download a FLEURS language from Hugging Face into FLAC clips + a manifest.

  python scripts/get_fleurs.py ur_pk --splits test
  python scripts/get_fleurs.py ur_pk hi_in --splits train dev test

Audio is streamed straight out of the tar archive and stored as FLAC, so the
download never sits on disk twice.
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.text import clean_label  # noqa: E402

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
# FLEURS config -> Whisper language code, where the prefix is not already the code.
WHISPER_CODE = {"cmn_hans_cn": "zh", "yue_hant_hk": "yue", "fil_ph": "tl", "nb_no": "no", "ast_es": None, "ckb_iq": None}


def whisper_lang(cfg):
    return WHISPER_CODE.get(cfg, cfg.split("_")[0])


def fetch_split(cfg, split, out_dir, max_clips=None):
    lang = whisper_lang(cfg)
    if lang is None:
        raise SystemExit(f"{cfg} is not a language Whisper supports")
    rows = {}
    tsv = requests.get(f"{BASE}/{cfg}/{split}.tsv", timeout=60)
    tsv.raise_for_status()
    for line in tsv.text.splitlines():
        cols = line.split("\t")
        rows[cols[1]] = {"sentence_id": cols[0], "text": cols[2], "samples": int(cols[5]), "gender": cols[6]}

    clip_dir = out_dir / "clips" / split
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"{split}.jsonl"
    print(f"{cfg}/{split}: {len(rows)} clips", flush=True)
    n = 0
    with requests.get(f"{BASE}/{cfg}/audio/{split}.tar.gz", stream=True, timeout=60) as resp, open(manifest, "w", encoding="utf-8") as out:
        resp.raise_for_status()
        resp.raw.decode_content = True
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in rows:
                    continue
                audio, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                dst = clip_dir / (Path(name).stem + ".flac")
                sf.write(dst, audio, sr)
                r = rows[name]
                rec = {
                    "id": f"fleurs-{cfg}-{Path(name).stem}",
                    "audio": str(dst.relative_to(out_dir)),  # relative to the manifest's folder
                    "text": clean_label(r["text"], lang),
                    "language": lang,
                    "duration": round(len(audio) / sr, 2),
                    "group": f"fleurs-{r['sentence_id']}",
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if max_clips and n >= max_clips:
                    break  # stop streaming: saves download time and disk
                if n % 200 == 0:
                    print(f"  {n}", flush=True)
    print(f"  wrote {n} clips -> {manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="FLEURS configs, e.g. ur_pk hi_in pa_in")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--max-clips", type=int, default=None, help="keep only the first N clips of each split")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "fleurs"))
    args = ap.parse_args()
    for cfg in args.configs:
        for split in args.splits:
            fetch_split(cfg, split, Path(args.data_dir) / cfg, args.max_clips)


if __name__ == "__main__":
    main()

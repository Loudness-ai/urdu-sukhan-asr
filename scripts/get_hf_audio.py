"""Download an audio dataset stored as parquet on Hugging Face into 16 kHz FLAC clips + a manifest.

  python scripts/get_hf_audio.py nzgnzg73/Urdu_Audio-Realistic --config Urdu-v1 --name nzg73
  python scripts/get_hf_audio.py mahwizzzz/UAT --name uat --no-text      # audio only; label it with pseudo_label.py

Writes data/<name>/clips/*.flac and data/<name>/all.jsonl. Shards are fetched one at a time and
deleted once their clips are written, so a dataset never sits on disk twice. Rerun to resume.
"""

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.text import clean_label  # noqa: E402

SR = 16000


def shard_urls(repo, config, exclude=(), shuffle=False):
    from huggingface_hub import list_repo_files  # works for gated datasets once logged in (hf auth login)

    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    if config:
        files = [f for f in files if f.startswith(config + "/")]
    files = [f for f in files if not any(x in f for x in exclude)]
    if shuffle:
        random.Random(0).shuffle(files)
    if not files:
        raise SystemExit(f"no parquet files for {repo} {config or ''}")
    return [(f, f"https://huggingface.co/datasets/{repo}/resolve/main/{f}") for f in files]


def download(url, dst):
    """Plain HTTP with resume (the hub client has stalled on big files here)."""
    from huggingface_hub import get_token

    part = dst.with_suffix(".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    if get_token():  # gated datasets
        headers["Authorization"] = f"Bearer {get_token()}"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            part.rename(dst)
            return dst
        r.raise_for_status()
        with open(part, "ab" if r.status_code == 206 else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    part.rename(dst)
    return dst


def columns(schema):
    audio = next((f.name for f in schema if str(f.type).startswith("struct") and "bytes" in str(f.type)), None)
    text = next((n for n in ("text", "sentence", "transcription", "transcript") if n in schema.names), None)
    if not audio:
        raise SystemExit(f"no audio column in {schema.names}")
    return audio, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--config", default=None, help="subfolder of the parquet files, if the repo has several")
    ap.add_argument("--name", required=True, help="output folder under data/")
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--no-text", action="store_true", help="keep audio only (transcripts unusable)")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--min-seconds", type=float, default=0.3)
    ap.add_argument("--exclude", nargs="*", default=[], help="skip parquet files whose path contains any of these")
    ap.add_argument("--per-shard", type=int, default=None, help="take at most N random clips from each file (spreads over speakers)")
    ap.add_argument("--shuffle", action="store_true", help="visit the files in random order")
    args = ap.parse_args()

    out = ROOT / "data" / args.name
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    manifest = out / "all.jsonl"
    seen = set()
    if manifest.exists():
        seen = {json.loads(l)["id"] for l in manifest.read_text("utf-8").splitlines() if l.strip()}
    n = len(seen)
    skipped = 0
    with open(manifest, "a", encoding="utf-8") as man:
        for fname, url in shard_urls(args.repo, args.config, args.exclude, args.shuffle):
            if args.max_clips and n >= args.max_clips:
                break
            shard = Path(fname).stem
            if (out / (shard + ".done")).exists():
                continue
            local = download(url, out / (shard + ".parquet"))
            pf = pq.ParquetFile(local)
            audio_col, text_col = columns(pf.schema_arrow)
            take = None
            if args.per_shard:  # a random subset of row numbers across the whole file
                total = pf.metadata.num_rows
                take = set(random.Random(shard).sample(range(total), min(args.per_shard, total)))
            offset = 0
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=[c for c in (audio_col, text_col) if c])
                rows_rg = table.to_pylist()
                for i, row in enumerate(rows_rg):
                    if take is not None and offset + i not in take:
                        continue
                    cid = f"{shard}-{rg}-{i}"
                    if cid in seen:
                        continue
                    if args.max_clips and n >= args.max_clips:
                        break
                    a = row[audio_col]
                    try:
                        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                    except Exception:
                        skipped += 1
                        continue
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    if sr != SR:
                        wav = soxr.resample(wav, sr, SR)
                    dur = len(wav) / SR
                    text = "" if args.no_text else clean_label((row.get(text_col) or "") if text_col else "", args.lang)
                    if dur < args.min_seconds or dur > args.max_seconds or (not args.no_text and not text):
                        skipped += 1
                        continue
                    sf.write(clips / f"{cid}.flac", np.clip(wav, -1, 1), SR)
                    rec = {"id": f"{args.name}-{cid}", "audio": f"clips/{cid}.flac", "text": text,
                           "language": args.lang, "duration": round(dur, 2), "group": args.repo}
                    man.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen.add(cid)
                    n += 1
                offset += len(rows_rg)
                man.flush()
            local.unlink()
            (out / (shard + ".done")).touch()
            print(f"{fname}: {n} clips so far, {skipped} skipped", flush=True)
    hours = sum(json.loads(l)["duration"] for l in manifest.read_text("utf-8").splitlines() if l.strip()) / 3600
    print(f"done: {n} clips, {hours:.1f} h -> {manifest}")


if __name__ == "__main__":
    main()

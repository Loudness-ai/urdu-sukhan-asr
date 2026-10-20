"""Score whisper.cpp models on a manifest: accuracy (WER/CER), speed, size.

  python scripts/evaluate.py data/fleurs/ur_pk/test.jsonl \
      --models small-q8_0 large-v3-turbo-q8_0 large-v3-q5_0 --name fleurs_ur_baseline

Models are stock names (downloaded once) or paths to our own ggml .bin files.
Writes results/<name>.md (table) and results/<name>/<model>.jsonl (every transcript).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.text import word_char_errors  # noqa: E402
from wlite.whispercpp import ensure_model, transcribe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_manifest(path, limit=None):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    for r in rows:
        r["audio_path"] = str((path.parent / r["audio"]).resolve())
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--rescore", action="store_true",
                    help="re-score saved transcripts under --name (e.g. after changing text rules) without running models")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lang", default=None, help="defaults to the manifest's language")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for whisper.cpp")
    ap.add_argument("--no-timestamps", action="store_true", help="decode without timestamp tokens, as the fine-tunes were trained")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.limit)
    lang = args.lang or rows[0]["language"]
    audio_s = sum(r["duration"] for r in rows)
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        refs = {r["id"]: r["text"] for r in rows}
        for sf in sorted(out_dir.glob("*.summary.json")):
            d = json.loads(sf.read_text())
            saved = [json.loads(l) for l in (out_dir / f"{d['model']}.jsonl").read_text("utf-8").splitlines() if l.strip()]
            old = (d["wer"], d["cer"])
            d["wer"], d["cer"] = word_char_errors([refs[x["id"]] for x in saved], [x["hyp"] for x in saved], lang)
            sf.write_text(json.dumps(d))
            print(f"{d['model']}: WER {old[0]:.1f} -> {d['wer']:.1f}%   CER {old[1]:.1f} -> {d['cer']:.1f}%")

    for m in args.models:
        path = ensure_model(m)
        label = Path(m).stem.removeprefix("ggml-") if m.endswith(".bin") else m
        print(f"\n== {label} on {len(rows)} clips ({audio_s / 60:.1f} min of audio)", flush=True)
        res, wall = transcribe(path, [r["audio_path"] for r in rows], lang, beam_size=args.beam_size, no_timestamps=args.no_timestamps, threads=args.threads)
        hyps = [res[r["audio_path"]]["text"] for r in rows]
        wer, cer = word_char_errors([r["text"] for r in rows], hyps, lang)
        with open(out_dir / f"{label}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
        size_mb = path.stat().st_size / 1e6
        summary = {"model": label, "size_mb": size_mb, "wer": wer, "cer": cer, "speed": audio_s / wall}
        (out_dir / f"{label}.summary.json").write_text(json.dumps(summary))
        print(f"   WER {wer:.1f}%  CER {cer:.1f}%  speed {audio_s / wall:.1f}x real time  ({size_mb:.0f} MB)", flush=True)

    lines = [
        f"# {args.name}",
        "",
        f"Manifest: `{args.manifest}` — {len(rows)} clips, {audio_s / 60:.1f} min, language `{lang}`.",
        f"whisper.cpp, beam size {args.beam_size}, Apple GPU (Metal). Speed includes model loading.",
        "CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.",
        "",
        "| Model | Size | WER % | CER % | Speed (x real time) |",
        "|---|---:|---:|---:|---:|",
    ]
    # The table covers every model scored under this name, including earlier runs.
    rows_done = sorted((json.loads(f.read_text()) for f in out_dir.glob("*.summary.json")), key=lambda d: d["size_mb"])
    lines += [f"| {d['model']} | {d['size_mb']:.0f} MB | {d['wer']:.1f} | {d['cer']:.1f} | {d['speed']:.1f} |" for d in rows_done]
    report = ROOT / "results" / f"{args.name}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[6:]))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()

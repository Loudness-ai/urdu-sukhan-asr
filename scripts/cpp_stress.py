"""Stress-test whisper.cpp models the way the app runs them: loops, invented text on no-speech clips, WER.

  python scripts/cpp_stress.py --models models/ggml/ggml-blend-r4r5-q8_0.bin models/ggml/ggml-r6c2775-q8_0.bin \
      --manifests data/stress/noise.jsonl data/stress/quiet.jsonl --name cpp_stress --threads 16

Clips marked "auto" are decoded with language detection, like the live page; the rest in their own language.
Writes results/<name>.md and results/<name>/<model>__<test>.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hf import read, runaway  # noqa: E402
from wlite.text import word_char_errors  # noqa: E402
from wlite.whispercpp import transcribe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--vad", default=None, help="Silero VAD model: skip audio without speech first, like whisper-cli --vad")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--vad-min-silence-ms", type=int, default=100, help="only split speech at pauses at least this long")
    ap.add_argument("--vad-pad-ms", type=int, default=30, help="keep this much audio around each speech part")
    args = ap.parse_args()

    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    extra = ["--vad", "-vm", args.vad, "-vt", str(args.vad_threshold), "-vsd", str(args.vad_min_silence_ms),
             "-vp", str(args.vad_pad_ms)] if args.vad else []
    for spec in args.manifests:
        rows = read(spec)
        test = Path(spec.partition(":")[0]).parent.name + "/" + Path(spec.partition(":")[0]).stem
        gen_lang = "auto" if rows[0]["language"].startswith("auto") else rows[0]["language"]
        score_lang = rows[0]["language"].partition(":")[2] or ("ur" if gen_lang == "auto" else gen_lang)
        for m in args.models:
            label = Path(m).stem.removeprefix("ggml-")
            res, wall = transcribe(m, [r["audio_path"] for r in rows], gen_lang, beam_size=args.beam_size,
                                   threads=args.threads, no_timestamps=True, progress=False, extra=extra)
            hyps = [res[r["audio_path"]]["text"].strip() for r in rows]
            with open(out / f"{label}__{test.replace('/', '_')}.jsonl", "w", encoding="utf-8") as f:
                for r, h in zip(rows, hyps):
                    f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
            speech = [(r["text"], h) for r, h in zip(rows, hyps) if r["text"].strip()]
            silent = [h for r, h in zip(rows, hyps) if not r["text"].strip()]
            cell = []
            if speech:
                wer, cer = word_char_errors([s for s, _ in speech], [h for _, h in speech], score_lang)
                cell.append(f"{wer:.1f} / {cer:.1f}")
            cell.append(f"loops {sum(runaway(h, s) for s, h in speech) + sum(len(h.split()) > 12 for h in silent)}")
            if silent:
                cell.append(f"invented {sum(len(h.split()) >= 2 for h in silent)}/{len(silent)}")
            table.setdefault(f"{test} ({len(rows)})", {})[label] = " · ".join(cell)
            print(f"{label:16s} {test}: {' · '.join(cell)}  ({wall:.0f}s)", flush=True)
    labels = [Path(m).stem.removeprefix("ggml-") for m in args.models]
    lines = [f"# {args.name}", "", "whisper.cpp, beam 5 with temperature fallback, no timestamps (as the app runs). "
             "Each cell: word / letter error rate, loops, invented text on no-speech clips.", "",
             "| Test set | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    lines += [f"| {t} | " + " | ".join(row.get(l, "") for l in labels) + " |" for t, row in table.items()]
    (ROOT / "results" / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""whisper.cpp stress test with any whisper-cli build and extra flags; counts one-word answers on no-speech too.

  python scripts/cpp_stress_x.py --models models/ggml/ggml-r6c2775-q8_0.bin --name cpp_teacher_nsb \
      --cli ~/src/whisper.cpp-nsb/build/bin --extra=-nsb --threads 12

Same decoding as scripts/cpp_stress.py (beam 5, temperature fallback, no timestamps, -l auto for "auto" sets).
--cli puts another whisper-cli first on PATH (e.g. a build with -nsb / --no-suppress-blank, which lets the model
end at once: by default whisper.cpp forbids end-of-text and blank as the first token, so every no-speech clip gets
at least one word). Each no-speech cell reports invented text (2+ words, as cpp_stress.py) and any word at all.
"""

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hf import read, runaway  # noqa: E402
from wlite.text import word_char_errors  # noqa: E402
from wlite.whispercpp import transcribe  # noqa: E402

STRESS = ["noise", "music", "silence", "quiet", "phone", "short", "english", "mixed", "long"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifests", nargs="*", default=[f"data/stress/{s}.jsonl" for s in STRESS])
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cli", help="folder holding the whisper-cli to use")
    ap.add_argument("--extra", default="", help="more whisper-cli flags, e.g. '-nsb' or the --vad flags")
    args = ap.parse_args()

    if args.cli:
        os.environ["PATH"] = str(Path(args.cli).expanduser()) + os.pathsep + os.environ["PATH"]
    extra = shlex.split(args.extra)
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    table, speed = {}, {}
    # whisper-cli writes <audio>.json next to each input and the reader deletes it, so two jobs on the same files
    # would eat each other's results: each job works on its own copy of the audio.
    work = Path(tempfile.mkdtemp(prefix=f"{args.name}-"))
    for spec in args.manifests:
        rows = read(spec)
        for k, r in enumerate(rows):
            dst = work / f"{Path(spec).stem}-{k}{Path(r['audio_path']).suffix}"
            shutil.copy(r["audio_path"], dst)
            r["audio_path"] = str(dst)
        test = Path(spec).parent.name + "/" + Path(spec).stem
        gen_lang = "auto" if rows[0]["language"].startswith("auto") else rows[0]["language"]
        score_lang = rows[0]["language"].partition(":")[2] or ("ur" if gen_lang == "auto" else gen_lang)
        audio_s = sum(float(r.get("duration") or 0) for r in rows)
        for m in args.models:
            label = Path(m).stem.removeprefix("ggml-")
            res, wall = transcribe(m, [r["audio_path"] for r in rows], gen_lang, beam_size=5, threads=args.threads,
                                   no_timestamps=True, progress=False, extra=extra)
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
                cell.append(f"any word {sum(len(h.split()) >= 1 for h in silent)}/{len(silent)}")
            table.setdefault(f"{test} ({len(rows)})", {})[label] = " · ".join(cell)
            s = speed.setdefault(label, [0.0, 0.0])
            s[0] += wall
            s[1] += audio_s
            print(f"{label:20s} {test}: {' · '.join(cell)}  ({wall:.0f}s for {audio_s:.0f}s of audio)", flush=True)
    labels = [Path(m).stem.removeprefix("ggml-") for m in args.models]
    lines = [f"# {args.name}", "", f"whisper.cpp, beam 5 with temperature fallback, no timestamps, {args.threads} threads"
             + (f", extra flags `{args.extra}`" if args.extra else "") + (f", whisper-cli from {args.cli}" if args.cli else "")
             + ". Each cell: word / letter error rate, loops, invented text (2+ words) and any word on no-speech clips.", "",
             "| Test set | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    lines += [f"| {t} | " + " | ".join(row.get(l, "") for l in labels) + " |" for t, row in table.items()]
    lines += ["", "Wall time / audio time (lower is faster): " + ", ".join(f"{l} {w / a:.2f}x" for l, (w, a) in speed.items() if a)]
    (ROOT / "results" / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

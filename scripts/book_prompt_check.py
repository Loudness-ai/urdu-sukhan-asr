"""Does a name prompt help whisper.cpp spell a book's names? Book-drafting trick only, not for the general app.

  python scripts/book_prompt_check.py --models models/ggml/ggml-r6c2775-q8_0.bin models/ggml/ggml-student-e16d4-q8_0.bin \
      --name book2_prompt --threads 8

Runs data/proofread/novel/test_book2.jsonl through whisper.cpp (beam 5, fallback, no timestamps, as the app)
with and without --prompt listing the names, and reports words wrong, loops, invented text and names spelled
right. Writes results/<name>/<model>__<mode>.jsonl and results/<name>.md.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hf import read, runaway  # noqa: E402
from name_check import BOOK2, count  # noqa: E402
from wlite.text import normalize_for_eval, word_char_errors  # noqa: E402
from wlite.whispercpp import transcribe  # noqa: E402

PROMPT = "سعدیہ راجپوت۔ نور الہدیٰ، اظہر فاروقی، عمیر، عزیر، تانیا، فائزہ، شایان، خدیجہ، ملیحہ، طارق۔"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default="data/proofread/novel/test_book2.jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    rows = read(args.manifest)
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    names = [normalize_for_eval(n, "ur") for n in BOOK2]
    lines = [f"# {args.name}", "", f"whisper.cpp beam 5, fallback, no timestamps, on {args.manifest}; prompt: {args.prompt}", "",
             "| model | prompt | WER / CER | loops | names right |", "|---|---|---:|---:|---:|"]
    for m in args.models:
        label = Path(m).stem.removeprefix("ggml-")
        for mode, prompt in (("no prompt", None), ("name prompt", args.prompt)):
            res, wall = transcribe(m, [r["audio_path"] for r in rows], "ur", beam_size=5, threads=args.threads,
                                   prompt=prompt, no_timestamps=True, progress=False)
            hyps = [res[r["audio_path"]]["text"].strip() for r in rows]
            with open(out / f"{label}__{mode.replace(' ', '_')}.jsonl", "w", encoding="utf-8") as f:
                for r, h in zip(rows, hyps):
                    f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
            wer, cer = word_char_errors([r["text"] for r in rows], hyps, "ur")
            loops = sum(runaway(h, r["text"]) for h, r in zip(hyps, rows))
            right = said = 0
            for r, h in zip(rows, hyps):
                ref, hyp = normalize_for_eval(r["text"], "ur"), normalize_for_eval(h, "ur")
                for n in names:
                    k = count(n, ref)
                    right += min(k, count(n, hyp))
                    said += k
            lines.append(f"| {label} | {mode} | {wer:.1f} / {cer:.1f} | {loops} | {right}/{said} |")
            print(lines[-1], f"({wall:.0f}s)", flush=True)
    (ROOT / "results" / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

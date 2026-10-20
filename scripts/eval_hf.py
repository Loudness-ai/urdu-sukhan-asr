"""Score Hugging Face Whisper models (stock or + LoRA adapters) on test manifests, on a CUDA GPU.
whisper.cpp scoring (scripts/evaluate.py) is the one that matches the app; this one is for the GPU
server, where whisper.cpp isn't installed and the test audio lives.

  python scripts/eval_hf.py --base models/hf/whisper-large-v3-turbo \
      --models stock r3=models/runs/novel-r3/adapter r4=models/runs/novel-r4/adapter \
      --manifests data/cv20k/test_balanced.jsonl data/fleurs/ur_pk/test.jsonl --name r4_check
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.audio import load_audio  # noqa: E402
from wlite.text import clean_label, word_char_errors  # noqa: E402


MARKS = re.compile(r"[ً-ٰٟ]")  # zer, zabar, pesh, shadd...: ignored when comparing repeats


def repeats(words, i, n):
    """How many times words[i:i+n] occurs back to back from position i."""
    k = 1
    while words[i + k * n : i + (k + 1) * n] == words[i : i + n]:
        k += 1
    return k


def runaway(hyp, ref):
    """Looping output: more than twice as many words as said, or the same 1-6 words four or more times in
    a row when the speaker didn't say them that often ("جی جی جی" in the reference is speech, not a loop)."""
    w, r = MARKS.sub("", hyp).split(), MARKS.sub("", ref).split()
    if len(w) > 2 * max(3, len(r)):
        return True
    for n in range(1, 7):
        for i in range(0, len(w) - 4 * n + 1):
            k = repeats(w, i, n)
            if k >= 4:
                said = max([repeats(r, j, n) for j in range(len(r) - n + 1) if r[j : j + n] == w[i : i + n]] or [0])
                if k > said + 1:
                    return True
    return False


def read(spec):
    path, _, n = spec.partition(":")
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    rows = rows[: int(n)] if n else rows
    for r in rows:
        r["audio_path"] = str((path.parent / r["audio"]).resolve())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="'stock', name=adapter_path or name=model_folder")
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--normalize", action="store_true", help="scale each clip to a common peak level first, like the app should")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.base)
    data = {m: read(m) for m in args.manifests}
    feats = {}
    for m, rows in data.items():  # audio features once, shared by every model
        def audio(r):
            a = load_audio(r["audio_path"])[: 30 * 16000]
            return a * (0.9 / max(float(abs(a).max()), 1e-4)) if args.normalize else a
        feats[m] = [processor.feature_extractor(audio(r), sampling_rate=16000, return_tensors="pt").input_features[0]
                    for r in rows]
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        if path and (Path(path) / "config.json").exists():  # a fully trained model folder
            model = WhisperForConditionalGeneration.from_pretrained(path, dtype=dtype)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=dtype)
            if path:
                model = PeftModel.from_pretrained(model, path).merge_and_unload()
        model.to(device).eval()
        for m, rows in data.items():
            hyps, t0 = [], time.time()
            for i in range(0, len(rows), args.batch):
                x = torch.stack(feats[m][i : i + args.batch]).to(device, dtype)
                lang_arg = rows[0]["language"]  # "auto" or "auto:en": let the model detect it, like the app does
                with torch.no_grad():
                    ids = model.generate(input_features=x, language=None if lang_arg.startswith("auto") else lang_arg,
                                         task="transcribe", max_new_tokens=400, num_beams=args.beams)
                hyps += [h.strip() for h in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]
            spec_lang = rows[0]["language"]  # scoring rules: "auto" -> Urdu, "auto:en" -> English
            lang = spec_lang.split(":", 1)[1] if ":" in spec_lang else ("ur" if spec_lang == "auto" else spec_lang)
            refs = [clean_label(r["text"], lang) for r in rows]
            speech = [i for i, r in enumerate(refs) if r]
            wer = cer = float("nan")
            if speech:
                wer, cer = word_char_errors([refs[i] for i in speech], [hyps[i] for i in speech], lang)
            loops = sum(runaway(h, r) for h, r in zip(hyps, refs))
            silent = [i for i, r in enumerate(refs) if not r]
            invented = sum(len(hyps[i].split()) >= 2 for i in silent)  # words "heard" where nothing was said
            table.setdefault(m, {})[name] = (wer, cer, loops, invented, len(silent))
            extra = f"  loops {loops}/{len(rows)}" + (f"  invented text on {invented}/{len(silent)} no-speech clips" if silent else "")
            print(f"{name:8s} {Path(m).parent.name}/{Path(m).stem}: WER {wer:.1f}%  CER {cer:.1f}%{extra}  ({time.time() - t0:.0f}s)", flush=True)
            with open(out_dir / f"{name}__{Path(m).parent.name}_{Path(m).stem}.jsonl", "w", encoding="utf-8") as f:
                for r, h in zip(rows, hyps):
                    f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()
    names = [s.partition("=")[0] for s in args.models]
    lines = [f"# {args.name}", "", "Hugging Face generate, greedy unless --beams. Each cell: word / letter error rate (lower is better), "
             "then loops, then invented text on no-speech clips.", "",
             "| Test set | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for m, row in table.items():
        def cell(v):
            wer, cer, loops, inv, sil = v
            out = "–" if wer != wer else f"{wer:.1f} / {cer:.1f}"
            out += f" · loops {loops}"
            return out + (f" · invented {inv}/{sil}" if sil else "")
        lines.append(f"| {Path(m).parent.name}/{Path(m).stem} ({len(data[m])}) | " + " | ".join(cell(row[n]) for n in names) + " |")
    (ROOT / "results" / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Check Transformers weights rebuilt by scripts/ggml_to_hf.py, on the Mac (one GPU job at a time).

  # 1. Transformers greedy on Apple GPU (MPS, fp16), like eval_hf.py on the server: language from the manifest,
  #    no timestamps, max 400 new tokens. Writes results/<name>/hf__<set>.jsonl (hf32__ with --dtype float32:
  #    the same weights, fp32 arithmetic) and prints WER / CER / loops.
  python scripts/verify_hf_rebuild.py hf --model ../release/hf-weights --name hf_rebuild \
      --manifests data/cv20k/test_unique.jsonl data/proofread/novel/test_book1.jsonl data/proofread/novel/test_book2.jsonl

  # 2. whisper.cpp greedy (-bs 1, no fallback, -nt) on the first N clips of each set, then compared with step 1.
  python scripts/verify_hf_rebuild.py cpp --ggml models/ggml/ggml-r6c2775-q8_0.bin --name hf_rebuild --first 4 \
      --manifests ...
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hf import read, runaway  # noqa: E402
from wlite.audio import load_audio, write_wav  # noqa: E402
from wlite.text import clean_label, normalize_for_eval, word_char_errors  # noqa: E402
from wlite.whispercpp import whisper_cli  # noqa: E402


def set_name(m):
    return f"{Path(m).parent.name}_{Path(m).stem}"


def run_hf(args, out_dir):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = getattr(torch, args.dtype)
    tag = "hf" if args.dtype == "float16" else "hf32"
    if args.model.endswith(".bin"):  # a ggml file: the same weights ggml_to_hf.py writes, rebuilt in memory
        from ggml_to_hf import rebuild

        processor = WhisperProcessor.from_pretrained(args.base)
        weights, _ = rebuild(args.model, args.base)
        model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=dtype).to(args.device)
        params = dict(model.named_parameters())  # proj_out is tied to embed_tokens, so listed once
        replaced = 0
        for k in list(weights):
            params[k].data.copy_(torch.from_numpy(weights.pop(k)).to(dtype))
            replaced += 1
        print(f"rebuilt in memory from {args.model}: {replaced} of {len(params)} parameter tensors replaced", flush=True)
    else:
        processor = WhisperProcessor.from_pretrained(args.model)
        model, info = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype, output_loading_info=True)
        print(f"loading: missing {info['missing_keys']}, unexpected {info['unexpected_keys']}, "
              f"mismatched {info['mismatched_keys']}", flush=True)
    tied = model.proj_out.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr()
    print(f"proj_out tied to embed_tokens: {tied}", flush=True)
    model.to(args.device).eval()
    summary = {}
    for m in args.manifests:
        rows = read(m)
        feats = [processor.feature_extractor(load_audio(r["audio_path"])[: 30 * 16000], sampling_rate=16000,
                                             return_tensors="pt").input_features[0] for r in rows]
        hyps, t0 = [], time.time()
        for i in range(0, len(rows), args.batch):
            x = torch.stack(feats[i : i + args.batch]).to(args.device, dtype)
            with torch.no_grad():
                ids = model.generate(input_features=x, language=rows[0]["language"], task="transcribe",
                                     max_new_tokens=400, num_beams=1)
            hyps += [h.strip() for h in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]
        lang = rows[0]["language"]
        refs = [clean_label(r["text"], lang) for r in rows]
        wer, cer = word_char_errors(refs, hyps, lang)
        loops = sum(runaway(h, r) for h, r in zip(hyps, refs))
        summary[set_name(m)] = {"clips": len(rows), "wer": wer, "cer": cer, "loops": loops}
        print(f"{set_name(m)} ({len(rows)}): WER {wer:.2f}%  CER {cer:.2f}%  loops {loops}  ({time.time() - t0:.0f}s)", flush=True)
        with open(out_dir / f"{tag}__{set_name(m)}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def run_cpp(args, out_dir):
    exe = whisper_cli()
    lines = []
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        for m in args.manifests:
            hf = {}  # "hf" (fp16) and "hf32" transcripts from the hf step, whichever exist
            for tag in ("hf", "hf32"):
                path = out_dir / f"{tag}__{set_name(m)}.jsonl"
                if path.exists():
                    hf[tag] = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in path.open(encoding="utf-8")}
            rows = read(m)
            rows = rows[: args.first] + [r for r in rows[args.first :] if r["id"] in args.ids]
            for k, r in enumerate(rows):
                wav = Path(tmp) / f"clip{k}.wav"
                write_wav(wav, load_audio(r["audio_path"])[: 30 * 16000])
                stem = Path(tmp) / f"clip{k}"
                cmd = [exe, "-m", args.ggml, "-l", r["language"], "-bs", "1", "-bo", "1", "-nf", "-nt", "-t", "4",
                       "-np", "-otxt", "-of", str(stem), "-f", str(wav)]
                subprocess.run(cmd, check=True, capture_output=True)
                cpp = " ".join(stem.with_suffix(".txt").read_text("utf-8", errors="replace").split())
                lang = r["language"]
                line = {"set": set_name(m), "id": r["id"], "cpp": cpp}
                for tag, hyps in hf.items():
                    h = hyps[r["id"]]
                    a, b = normalize_for_eval(clean_label(h, lang), lang), normalize_for_eval(clean_label(cpp, lang), lang)
                    same = "identical" if h.split() == cpp.split() else ("same after normalisation" if a == b else "differs")
                    line[tag] = {"text": h, "match": same, "word_diff_pct": word_char_errors([a], [b], lang)[0] if a else 0.0}
                lines.append(line)
                print(f"{set_name(m)} {r['id']}: " + ", ".join(f"{t} {line[t]['match']} ({line[t]['word_diff_pct']:.1f}% words)"
                                                          for t in hf), flush=True)
    with open(out_dir / "cpp_vs_hf.jsonl", "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["hf", "cpp"])
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", help="rebuilt Transformers folder, or the ggml file to rebuild in memory (hf step)")
    ap.add_argument("--base", default=str(ROOT / "models" / "hf" / "whisper-large-v3-turbo"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="arithmetic for the hf step")
    ap.add_argument("--ggml", help="whisper.cpp model (cpp step)")
    ap.add_argument("--first", type=int, default=4, help="clips per set for the cpp step")
    ap.add_argument("--ids", nargs="*", default=[], help="more clips for the cpp step, by id")
    ap.add_argument("--tmp", default=None, help="folder for temporary wav files")
    args = ap.parse_args()
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_hf(args, out_dir) if args.step == "hf" else run_cpp(args, out_dir)


if __name__ == "__main__":
    main()

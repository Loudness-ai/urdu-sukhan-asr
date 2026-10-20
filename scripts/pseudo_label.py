"""Label untranscribed audio with one of our models and keep only the confident results.

  # on a CUDA GPU (fast): the Hugging Face base model plus a LoRA adapter
  python scripts/pseudo_label.py data/uat/all.jsonl --out data/uat/labeled.jsonl \
      --base models/hf/whisper-large-v3-turbo --adapter models/runs/novel-r3/adapter
  # on the Mac: a whisper.cpp model
  python scripts/pseudo_label.py data/uat/all.jsonl --out data/uat/labeled.jsonl --ggml models/ggml/ggml-novel-r3-q8_0.bin

These labels are machine-made ("silver"): train on them with a lower weight than proofread clips.
A clip is dropped when the model is unsure, repeats itself, or the text is implausibly short or long
for the audio. Rerun to resume.
"""

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.text import clean_label  # noqa: E402


def keep(text, dur, conf, min_conf):
    """Why a label should be dropped, or None to keep it."""
    if not text:
        return "empty"
    letters = len(text.replace(" ", ""))
    if len(text) > 30 and len(text) / len(zlib.compress(text.encode("utf-8"))) > 2.4:
        return "repeats"
    if not 3 <= letters / max(dur, 0.1) <= 25:
        return "length"
    if conf < min_conf:
        return "unsure"
    return None


def label_hf(rows, base, adapter, batch):
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from wlite.audio import load_audio

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    processor = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        audio = [load_audio(r["audio_path"])[: 30 * 16000] for r in part]
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            out = model.generate(input_features=feats.to(device, dtype), language=part[0]["language"], task="transcribe",
                                 max_new_tokens=400, num_beams=1, return_dict_in_generate=True, output_scores=True)
        texts = processor.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        # Greedy decoding picks the most likely token at each step, so its probability is the max of
        # that step's distribution. (Whisper's generate strips the prompt from `sequences`, which throws
        # off compute_transition_scores' alignment.) Average up to and including end-of-text.
        eos = processor.tokenizer.eos_token_id
        total = torch.zeros(len(part), device=out.scores[0].device)
        count = torch.zeros(len(part), device=out.scores[0].device)
        alive = torch.ones(len(part), dtype=torch.bool, device=out.scores[0].device)
        for step in out.scores:
            lp = torch.log_softmax(step.float(), dim=-1)
            best, tok = lp.max(dim=-1)
            total += torch.where(alive, best, torch.zeros_like(best))
            count += alive.float()
            alive &= tok != eos
            if not alive.any():
                break
        for r, text, t, c in zip(part, texts, total.tolist(), count.tolist()):
            yield r, text.strip(), math.exp(t / c) if c else 0.0


def label_ggml(rows, ggml, batch):
    from wlite.whispercpp import transcribe

    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        res, _ = transcribe(ggml, [r["audio_path"] for r in part], part[0]["language"], batch=batch,
                            progress=False, no_timestamps=True)
        for r in part:
            words = res[r["audio_path"]]["words"]
            conf = math.exp(sum(math.log(max(w["p"], 1e-4)) for w in words) / len(words)) if words else 0.0
            yield r, res[r["audio_path"]]["text"], conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", help="Hugging Face base model (with --adapter for a LoRA run)")
    ap.add_argument("--adapter")
    ap.add_argument("--ggml", help="whisper.cpp model instead of Hugging Face")
    ap.add_argument("--min-conf", type=float, default=0.6, help="drop clips whose average token probability is lower")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.manifest)
    rows = [json.loads(l) for l in src.read_text("utf-8").splitlines() if l.strip()]
    for r in rows:
        r["audio_path"] = str((src.parent / r["audio"]).resolve())
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text("utf-8").splitlines() if l.strip()}
    rejected = out.with_suffix(".dropped.jsonl")
    if rejected.exists():
        done |= {json.loads(l)["id"] for l in rejected.read_text("utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    print(f"{len(todo)} clips to label ({len(done)} done before)", flush=True)

    gen = label_ggml(todo, args.ggml, args.batch) if args.ggml else label_hf(todo, args.base, args.adapter, args.batch)
    model_name = Path(args.ggml).stem if args.ggml else Path(args.adapter or args.base).parent.name
    counts = {"kept": 0}
    with open(out, "a", encoding="utf-8") as fo, open(rejected, "a", encoding="utf-8") as fr:
        for n, (r, text, conf) in enumerate(gen, 1):
            text = clean_label(text, r["language"])
            rec = {k: v for k, v in r.items() if k != "audio_path"}
            rec.update(text=text, conf=round(conf, 3), labeled_by=model_name)
            why = keep(text, r["duration"], conf, args.min_conf)
            if why:
                counts[why] = counts.get(why, 0) + 1
                fr.write(json.dumps({**rec, "dropped": why}, ensure_ascii=False) + "\n")
            else:
                counts["kept"] += 1
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                fo.flush(); fr.flush()
                print(f"  {n}/{len(todo)} {counts}", flush=True)
    print("done:", counts)


if __name__ == "__main__":
    main()

"""Merge a LoRA adapter into its base model and convert it for whisper.cpp (CPU, Metal, iOS).

  python scripts/export_ggml.py --base models/hf/whisper-large-v3-turbo \
      --adapter models/runs/novel-r1/adapter --name novel-r1

Writes models/ggml/ggml-<name>-q8_0.bin. Uses whisper.cpp's own converter (pinned to the
release Homebrew installs) and its quantize tool, so the file loads exactly like the stock ones.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
WHISPER_CPP_TAG = "v1.9.5"
CONVERTER = f"https://raw.githubusercontent.com/ggml-org/whisper.cpp/{WHISPER_CPP_TAG}/models/convert-h5-to-ggml.py"
MEL_FILTERS = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
TOKENIZER_FILES = ["vocab.json", "added_tokens.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "normalizer.json", "preprocessor_config.json", "generation_config.json"]


def fetch(url, dst):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", help="LoRA adapter to merge into --base")
    ap.add_argument("--model", help="an already fully trained model folder (instead of --adapter)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--quant", nargs="+", default=["q8_0"], help="one or more of q8_0 (near-lossless), q5_0, q4_0, f16")
    ap.add_argument("--keep-merged", action="store_true", help="keep the merged Hugging Face model")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    work = ROOT / "models" / "runs" / f"{args.name}-merged"
    if args.model:
        base = model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
    elif args.adapter:
        print("merging adapter ...", flush=True)
        base = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    else:
        raise SystemExit("give --adapter or --model")
    # float16: whisper.cpp's converter reads weights through numpy, which has no bfloat16.
    model.half().save_pretrained(work)
    for f in TOKENIZER_FILES:
        if (Path(args.base) / f).exists():
            shutil.copy(Path(args.base) / f, work / f)
    del model, base

    converter = fetch(CONVERTER, TOOLS / "whispercpp" / "convert-h5-to-ggml.py")
    fetch(MEL_FILTERS, TOOLS / "openai-whisper" / "whisper" / "assets" / "mel_filters.npz")
    out_dir = ROOT / "models" / "ggml"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("converting to ggml ...", flush=True)
    subprocess.run([sys.executable, str(converter), str(work), str(TOOLS / "openai-whisper"), str(work)],
                   check=True, stdout=subprocess.DEVNULL)
    f16 = work / "ggml-model.bin"
    quantize = shutil.which("whisper-quantize")
    for q in args.quant:
        final = out_dir / f"ggml-{args.name}-{q}.bin"
        if q == "f16":
            shutil.copy(f16, final)
        else:
            if not quantize:
                raise SystemExit("whisper-quantize not found (brew install whisper-cpp)")
            subprocess.run([quantize, str(f16), str(final), q], check=True, stdout=subprocess.DEVNULL)
        print(f"done: {final} ({final.stat().st_size / 1e6:.0f} MB)")
    if not args.keep_merged:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()

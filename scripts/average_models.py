"""Blend fine-tuned Whisper models by averaging their weights ("model soup").

Two fine-tunes of the same base often average into a model that keeps the gains of both and the
quirks of neither (for example one that's more accurate and one that never loops).

  python scripts/average_models.py --base models/hf/whisper-large-v3-turbo \
      --models models/runs/novel-r4/adapter models/runs/novel-r5/model --weights 0.5 0.5 \
      --out models/runs/soup-r4r5/model

Each --models entry is a LoRA adapter folder (merged into --base first) or a full model folder.
"""

import argparse
from pathlib import Path

import torch


def load(path, base):
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    if (Path(path) / "config.json").exists():
        return WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float32)
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
    return PeftModel.from_pretrained(model, path).merge_and_unload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(weights) == len(args.models) and abs(sum(weights) - 1) < 1e-6, "weights must match --models and sum to 1"

    from transformers import WhisperProcessor

    avg = None
    for path, w in zip(args.models, weights):
        state = load(path, args.base).state_dict()
        if avg is None:
            avg = {k: v.clone() * w for k, v in state.items()}
        else:
            for k, v in state.items():
                avg[k] += v * w
        del state
    model = load(args.models[0], args.base)
    model.load_state_dict(avg)
    model.half().save_pretrained(args.out)
    WhisperProcessor.from_pretrained(args.base).save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

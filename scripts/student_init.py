"""Make a smaller student Whisper from a trained teacher by keeping some of its layers.

  python scripts/student_init.py --teacher models/runs/novel-r6/ckpt-2775 --enc-layers 16 --dec-layers 4 \
      --out models/students/init-e16d4

Kept layers are spread evenly and always include the first and the last (the last encoder layer feeds
the decoder; the first sees the audio front-end). The decoder keeps its first and last layers when
shrunk to 2, as Distil-Whisper does. Everything else (front-end convolutions, embeddings, final layer
norms, tokenizer files) is copied unchanged. The result is a normal Hugging Face model folder.
"""

import argparse
import json
import shutil
from pathlib import Path


def spread(n, k):
    """k of n layer indices, evenly spaced, first and last included."""
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--enc-layers", type=int, required=True)
    ap.add_argument("--dec-layers", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration

    teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher, dtype=torch.float32)
    cfg = teacher.config
    enc = spread(cfg.encoder_layers, args.enc_layers)
    dec = spread(cfg.decoder_layers, args.dec_layers)
    assert len(enc) == args.enc_layers and len(dec) == args.dec_layers, (enc, dec)
    print(f"encoder layers kept {enc}\ndecoder layers kept {dec}", flush=True)

    m = teacher.model
    m.encoder.layers = torch.nn.ModuleList([m.encoder.layers[i] for i in enc])
    m.decoder.layers = torch.nn.ModuleList([m.decoder.layers[i] for i in dec])
    for j, layer in enumerate(m.decoder.layers):  # attention modules carry their position for the KV cache
        for att in (layer.self_attn, layer.encoder_attn):
            if hasattr(att, "layer_idx"):
                att.layer_idx = j
    for j, layer in enumerate(m.encoder.layers):
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = j
    cfg.encoder_layers, cfg.decoder_layers = len(enc), len(dec)
    cfg.num_hidden_layers = len(enc)
    teacher.config = cfg

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = teacher.generation_config
    # Word-timing heads: the teacher's sit in decoder layers 2-3. Keep the ones whose layer survived,
    # renumbered; with a 2-layer decoder use every head of the last layer.
    old = getattr(gen, "alignment_heads", None) or []
    heads = [[dec.index(l), h] for l, h in old if l in dec]
    gen.alignment_heads = heads or [[len(dec) - 1, h] for h in range(cfg.decoder_attention_heads)]
    teacher.save_pretrained(out, state_dict={k: v.half() for k, v in teacher.state_dict().items()})
    for f in Path(args.teacher).iterdir():  # tokenizer + feature extractor files
        if f.suffix in (".json", ".txt") and f.name not in ("config.json", "generation_config.json") and not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    (out / "student_init.json").write_text(json.dumps({"teacher": args.teacher, "encoder_layers": enc,
                                                       "decoder_layers": dec}, indent=2))
    n = sum(p.numel() for p in teacher.parameters())
    print(f"saved {out}: {n / 1e6:.0f}M parameters", flush=True)


if __name__ == "__main__":
    main()

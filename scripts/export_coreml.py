"""Core ML encoder for whisper.cpp on iOS/macOS (runs the encoder on the Neural Engine), from a Hugging Face model.

  ~/coreml-venv/bin/python scripts/export_coreml.py --model models/runs/student-e8d4/ckpt-3000 --name student-e8d4

Needs a separate environment with coremltools, openai-whisper and ane_transformers (see requirements-coreml.txt
in whisper.cpp; this was set up as ~/coreml-venv on the GPU server), and whisper.cpp's own converter in
~/src/whisper.cpp/models. Writes models/coreml/coreml-encoder-<name>.mlpackage (float16) plus a reference
input/output pair for checking it on a Mac:
  xcrun coremlc compile coreml-encoder-<name>.mlpackage .   ->  rename to ggml-<name>-encoder.mlmodelc
whisper.cpp built with WHISPER_COREML=1 loads ggml-<name>-encoder.mlmodelc from next to ggml-<name>-*.bin.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONVERTER = Path.home() / "src" / "whisper.cpp" / "models" / "convert-whisper-to-coreml.py"

# Hugging Face -> openai-whisper parameter names (as in whisper.cpp's convert-h5-to-coreml.py)
MAPPING = [
    ("layers", "blocks"), ("fc1", "mlp.0"), ("fc2", "mlp.2"), ("final_layer_norm", "mlp_ln"),
    (".self_attn.q_proj", ".attn.query"), (".self_attn.k_proj", ".attn.key"), (".self_attn.v_proj", ".attn.value"),
    (".self_attn_layer_norm", ".attn_ln"), (".self_attn.out_proj", ".attn.out"),
    (".encoder_attn.q_proj", ".cross_attn.query"), (".encoder_attn.k_proj", ".cross_attn.key"),
    (".encoder_attn.v_proj", ".cross_attn.value"), (".encoder_attn_layer_norm", ".cross_attn_ln"),
    (".encoder_attn.out_proj", ".cross_attn.out"), ("decoder.layer_norm.", "decoder.ln."),
    ("encoder.layer_norm.", "encoder.ln_post."), ("embed_tokens", "token_embedding"),
    ("encoder.embed_positions.weight", "encoder.positional_embedding"),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
]


def to_openai(hf_dir):
    from safetensors.torch import load_file
    from whisper.model import ModelDimensions, Whisper

    cfg = json.loads((Path(hf_dir) / "config.json").read_text())
    sd = load_file(str(Path(hf_dir) / "model.safetensors"))
    sd = {k.removeprefix("model."): v.float() for k, v in sd.items() if not k.startswith("proj_out")}
    out = {}
    for k, v in sd.items():
        for a, b in MAPPING:
            k = k.replace(a, b)
        out[k] = v
    dims = ModelDimensions(n_mels=cfg["num_mel_bins"], n_audio_ctx=cfg["max_source_positions"], n_audio_state=cfg["d_model"],
                           n_audio_head=cfg["encoder_attention_heads"], n_audio_layer=cfg["encoder_layers"],
                           n_vocab=cfg["vocab_size"], n_text_ctx=cfg["max_target_positions"], n_text_state=cfg["d_model"],
                           n_text_head=cfg["decoder_attention_heads"], n_text_layer=cfg["decoder_layers"])
    model = Whisper(dims)
    missing, unexpected = model.load_state_dict(out, strict=False)
    missing = [m for m in missing if m != "alignment_heads"]
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval(), dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model folder")
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="models/hf/whisper-large-v3-turbo", help="folder with preprocessor_config.json (128 mel bins)")
    ap.add_argument("--no-ane", action="store_true", help="plain layout instead of whisper.cpp's Neural Engine layout")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("whisper_to_coreml", CONVERTER)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    model, dims = to_openai(args.model)
    print(dims, flush=True)
    encoder = model.encoder
    if not args.no_ane:
        ane = conv.WhisperANE(dims).eval()
        ane.load_state_dict(model.state_dict())
        encoder = ane.encoder
    out = ROOT / "models" / "coreml"
    out.mkdir(parents=True, exist_ok=True)

    # Reference pair: a real clip's log-mel and the PyTorch encoder output (float32), for checking on a Mac.
    from transformers import WhisperFeatureExtractor
    from wlite.audio import load_audio
    clip = json.loads((ROOT / "data/stress/phone.jsonl").read_text("utf-8").splitlines()[0])
    fe = WhisperFeatureExtractor.from_pretrained(args.features)
    mel = fe(load_audio(str(ROOT / "data/stress" / clip["audio"])), sampling_rate=16000, return_tensors="np").input_features
    with torch.no_grad():
        ref = model.encoder(torch.from_numpy(mel)).numpy()
        ane_out = encoder(torch.from_numpy(mel)).numpy()
    if ane_out.ndim == 4:  # ANE layout: (batch, channels, 1, frames)
        ane_out = ane_out.squeeze(2).transpose(0, 2, 1)
    print(f"PyTorch check, ANE layout vs plain: max abs diff {np.abs(ane_out - ref).max():.2e}", flush=True)
    np.save(out / f"{args.name}-ref-mel.npy", mel.astype(np.float32))
    np.save(out / f"{args.name}-ref-out.npy", ref.astype(np.float32))

    ml = conv.convert_encoder(dims, encoder, quantize=True)  # float16 compute, as the Neural Engine runs
    path = out / f"coreml-encoder-{args.name}.mlpackage"
    ml.save(str(path))
    print(f"done: {path}", flush=True)


if __name__ == "__main__":
    main()

"""Rebuild Hugging Face Transformers weights (fp16 safetensors) from a whisper.cpp ggml file.

  python scripts/ggml_to_hf.py --ggml models/ggml/ggml-r6c2775-q8_0.bin \
      --base models/hf/whisper-large-v3-turbo --out ../release/hf-weights

The inverse of whisper.cpp's convert-h5-to-ggml.py (tools/whispercpp/) followed by whisper-quantize:
tensor names are mapped back, q8_0 blocks (32 int8 values sharing one fp16 scale) are dequantized, f16/f32
tensors are copied, the conv biases lose their extra [n, 1] axis, and proj_out stays tied to the token
embedding (the converter skips it). Every tensor must match the base model's name, shape and count, or the
script stops. config.json is the base's (use_cache true, apply_spec_augment false); the tokenizer,
preprocessor and generation files are copied from the base unchanged.

The f16 and f32 tensors come back exactly (the converter wrote them from fp16). The q8_0 ones carry
whisper.cpp's rounding error, so the result is the q8_0 model in fp16, not the lost original checkpoint.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

GGML_MAGIC = 0x67676D6C
F32, F16, Q8_0 = 0, 1, 8
QK8_0 = 32
BLOCK_Q8_0 = np.dtype([("d", "<f2"), ("qs", "i1", (QK8_0,))])  # 34 bytes

# whisper.cpp name -> Transformers name, the reverse of conv_map in convert-h5-to-ggml.py
LAYER_PARTS = {
    "attn.query": "self_attn.q_proj",
    "attn.key": "self_attn.k_proj",
    "attn.value": "self_attn.v_proj",
    "attn.out": "self_attn.out_proj",
    "attn_ln": "self_attn_layer_norm",
    "cross_attn.query": "encoder_attn.q_proj",
    "cross_attn.key": "encoder_attn.k_proj",
    "cross_attn.value": "encoder_attn.v_proj",
    "cross_attn.out": "encoder_attn.out_proj",
    "cross_attn_ln": "encoder_attn_layer_norm",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
}
TOP_LEVEL = {
    "encoder.ln_post.weight": "encoder.layer_norm.weight",
    "encoder.ln_post.bias": "encoder.layer_norm.bias",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.ln.weight": "decoder.layer_norm.weight",
    "decoder.ln.bias": "decoder.layer_norm.bias",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "decoder.token_embedding.weight": "decoder.embed_tokens.weight",
}
BLOCK = re.compile(r"(encoder|decoder)\.blocks\.(\d+)\.(.+)\.(weight|bias)")
COPY_FILES = ["added_tokens.json", "generation_config.json", "merges.txt", "normalizer.json",
              "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json"]
HPARAMS = ["vocab_size", "max_source_positions", "d_model", "encoder_attention_heads", "encoder_layers",
           "max_target_positions", "d_model", "decoder_attention_heads", "decoder_layers", "num_mel_bins"]


def hf_name(name):
    if name in TOP_LEVEL:
        return "model." + TOP_LEVEL[name]
    m = BLOCK.fullmatch(name)
    if m:
        side, i, part, kind = m.groups()
        return f"model.{side}.layers.{i}.{LAYER_PARTS[part]}.{kind}"
    return "model." + name  # encoder.conv1.weight etc. keep their names


def read_ggml(path):
    """Yield (ggml name, ggml type, numpy shape, float32 array) for every tensor, after checking the header."""
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    pos = 0

    def ints(n):
        nonlocal pos
        v = np.frombuffer(buf, "<i4", n, pos)
        pos += 4 * n
        return [int(x) for x in v]

    header = {"magic": ints(1)[0], "hparams": ints(10), "ftype": ints(1)[0]}
    n_mel, n_fft = ints(2)
    pos += 4 * n_mel * n_fft  # mel filter bank
    tokens = []
    for _ in range(ints(1)[0]):
        n = ints(1)[0]
        tokens.append(bytes(buf[pos : pos + n]))
        pos += n
    header["tokens"] = tokens
    yield header
    while pos < len(buf):
        n_dims, name_len, ttype = ints(3)
        ne = ints(n_dims)
        name = bytes(buf[pos : pos + name_len]).decode()
        pos += name_len
        count = int(np.prod(ne))
        shape = tuple(reversed(ne))  # ggml ne[0] is the fastest-moving axis
        if ttype == F32:
            data = np.frombuffer(buf, "<f4", count, pos).astype(np.float32)
            pos += 4 * count
        elif ttype == F16:
            data = np.frombuffer(buf, "<f2", count, pos).astype(np.float32)
            pos += 2 * count
        elif ttype == Q8_0:
            blocks = np.frombuffer(buf, BLOCK_Q8_0, count // QK8_0, pos)
            pos += BLOCK_Q8_0.itemsize * (count // QK8_0)
            data = (blocks["d"].astype(np.float32)[:, None] * blocks["qs"].astype(np.float32)).reshape(-1)
        else:
            raise SystemExit(f"{name}: ggml type {ttype} not handled (only f32, f16, q8_0)")
        yield name, ttype, shape, data.reshape(shape)


def rebuild(ggml_path, base_dir):
    """Return ({Transformers name: fp16 array}, per-tensor report), checked against the base model."""
    from safetensors import safe_open

    base_dir = Path(base_dir)
    config = json.loads((base_dir / "config.json").read_text())
    base = safe_open(base_dir / "model.safetensors", "np")
    base_keys = set(base.keys())

    tensors = read_ggml(ggml_path)
    header = next(tensors)
    if header["magic"] != GGML_MAGIC:
        raise SystemExit("not a ggml file")
    want = [config[k] for k in HPARAMS]
    if header["hparams"] != want:
        raise SystemExit(f"hyper-parameters {header['hparams']} differ from the base config {want}")
    vocab = json.loads((base_dir / "vocab.json").read_text("utf-8"))
    if len(header["tokens"]) != len(vocab):
        raise SystemExit(f"{len(header['tokens'])} tokens in the ggml file, {len(vocab)} in vocab.json")

    weights, report = {}, []
    for name, ttype, shape, data in tensors:
        key = hf_name(name)
        if key not in base_keys:
            raise SystemExit(f"{name} -> {key}: not in the base model")
        ref = base.get_tensor(key)
        if data.shape != ref.shape:
            if data.size == ref.size and name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(ref.shape)  # the converter stored them as [n, 1]
            else:
                raise SystemExit(f"{key}: shape {data.shape}, base {ref.shape}")
        w = data.astype(np.float16)
        if not np.isfinite(w).all():
            raise SystemExit(f"{key}: non-finite values after fp16 cast")
        weights[key] = w
        ref32 = ref.astype(np.float32)
        change = float(np.linalg.norm(w.astype(np.float32) - ref32) / max(np.linalg.norm(ref32), 1e-12))
        report.append({"name": key, "ggml": name, "type": {F32: "f32", F16: "f16", Q8_0: "q8_0"}[ttype],
                       "shape": list(w.shape), "change_from_base": change})
    missing = base_keys - set(weights)
    if missing:
        raise SystemExit(f"{len(missing)} base tensors missing from the ggml file, e.g. {sorted(missing)[:5]}")
    print(f"ggml header ftype {header['ftype']}")
    return weights, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ggml", required=True)
    ap.add_argument("--base", required=True, help="the Transformers model the ggml file was made from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors.numpy import save_file

    base_dir, out = Path(args.base), Path(args.out)
    config = json.loads((base_dir / "config.json").read_text())
    weights, report = rebuild(args.ggml, base_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})
    config["use_cache"] = True
    config["apply_spec_augment"] = False
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for f in COPY_FILES:
        shutil.copy(base_dir / f, out / f)
    (out / "conversion_report.json").write_text(json.dumps(report, indent=1) + "\n")

    types = {}
    for r in report:
        types[r["type"]] = types.get(r["type"], 0) + 1
    changes = [r["change_from_base"] for r in report]
    print(f"{len(weights)} tensors written ({types})")
    print(f"relative change from the base model: median {np.median(changes):.4f}, max {max(changes):.4f} "
          f"({max(report, key=lambda r: r['change_from_base'])['name']})")


if __name__ == "__main__":
    main()

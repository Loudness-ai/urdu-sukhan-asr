"""Run whisper.cpp (Metal on Mac) and turn its output into words with confidence and timing."""

import json
import os
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGML_DIR = ROOT / "models" / "ggml"
EOT = 50257  # first special token id in every multilingual Whisper vocab


def whisper_cli():
    exe = shutil.which("whisper-cli")
    if not exe:
        raise RuntimeError("whisper-cli not found. Install it with: brew install whisper-cpp")
    return exe


def ensure_model(name_or_path):
    """Accept a local .bin path, or a stock name like 'large-v3-turbo-q8_0' (downloaded once)."""
    p = Path(name_or_path)
    if p.suffix == ".bin" and p.exists():
        return p
    import requests

    GGML_DIR.mkdir(parents=True, exist_ok=True)
    dst = GGML_DIR / f"ggml-{name_or_path}.bin"
    if dst.exists():
        return dst
    # Plain HTTP with resume: more predictable than the hub client for one big file.
    part = dst.with_suffix(".part")
    have = part.stat().st_size if part.exists() else 0
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{dst.name}"
    with requests.get(url, stream=True, timeout=60, headers={"Range": f"bytes={have}-"} if have else {}) as r:
        r.raise_for_status()
        total = have + int(r.headers.get("Content-Length", 0))
        print(f"Downloading {dst.name} ({total / 1e6:.0f} MB) ...", flush=True)
        with open(part, "ab" if r.status_code == 206 else "wb") as f:
            done, last = (have if r.status_code == 206 else 0), 0
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done - last > 100e6:
                    print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                    last = done
    part.rename(dst)
    return dst


@lru_cache(maxsize=1)
def _id_to_bytes():
    """Token id -> raw bytes, from the byte-level BPE vocab shared by all multilingual Whisper models."""
    from huggingface_hub import hf_hub_download

    vocab = json.loads(Path(hf_hub_download("openai/whisper-small", "vocab.json")).read_text("utf-8"))
    # Inverse of GPT-2's bytes_to_unicode table.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    return {i: bytes(char_to_byte[ch] for ch in tok) for tok, i in vocab.items() if i < EOT}


def tokens_to_words(tokens):
    """Group whisper.cpp tokens into words. Each word gets min token probability and start/end ms."""
    id2b = _id_to_bytes()
    words = []
    for t in tokens:
        tid = t["id"]
        if tid >= EOT or tid not in id2b:
            continue
        # Recent whisper.cpp merges a letter split over two byte-tokens into the first token's
        # text and drops the second token, so prefer the text; fall back to the id's raw bytes
        # when the text itself is a broken fragment.
        text = t.get("text", "")
        b = text.encode("utf-8") if text and "�" not in text else id2b[tid]
        start, end = t["offsets"]["from"], t["offsets"]["to"]
        if t.get("t_dtw", -1) >= 0:  # attention-aligned time (centiseconds); the only good timing without timestamp tokens
            start = end = t["t_dtw"] * 10
        if not words or b.startswith(b" "):
            words.append({"b": b.lstrip(b" "), "p": t["p"], "start": start, "end": end})
        else:
            w = words[-1]
            w["b"] += b
            w["p"] = min(w["p"], t["p"])
            w["end"] = end
    if any(t.get("t_dtw", -1) >= 0 for t in tokens):
        # DTW gives one time per token, so a word runs from its first token to the next word's start.
        for w, nxt in zip(words, words[1:]):
            w["end"] = max(w["end"], nxt["start"])
        if words:
            words[-1]["end"] += 300
    out = []
    for w in words:
        text = w["b"].decode("utf-8", errors="replace").strip()
        if text:
            out.append({"w": text, "p": round(w["p"], 3), "start": w["start"], "end": w["end"]})
    return out


def transcribe(model, wav_paths, lang, beam_size=5, threads=4, batch=40, prompt=None, progress=True, no_timestamps=False, dtw="large.v3.turbo", extra=()):
    """Transcribe 16 kHz wav files (extra: more whisper-cli flags, e.g. the voice detector). Returns ({path: {"text", "words"}}, wall_seconds).

    Files are passed to one whisper-cli process per batch so the model loads once per batch.
    """
    exe = whisper_cli()
    model = str(ensure_model(model))
    results = {}
    wall = 0.0
    wav_paths = [str(p) for p in wav_paths]
    for i in range(0, len(wav_paths), batch):
        group = wav_paths[i : i + batch]
        cmd = [exe, "-m", model, "-l", lang, "-bs", str(beam_size), "-t", str(threads), "-ojf", "-np"]
        if prompt:
            cmd += ["--prompt", prompt]
        if no_timestamps:  # decode the way our fine-tunes were trained (<|notimestamps|>)
            # Without timestamp tokens whisper.cpp's word times drift, so align words with DTW
            # over cross-attention instead (needs flash attention off).
            cmd += ["-nt", "-nfa", "-dtw", dtw]
        cmd += list(extra)
        for p in group:
            cmd += ["-f", p]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True)
        wall += time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli failed:\n{proc.stderr.decode(errors='replace')[-2000:]}")
        for p in group:
            jpath = p + ".json"
            if not os.path.exists(jpath):
                results[p] = {"text": "", "words": []}
                continue
            # Tokens can split a UTF-8 character, so decode leniently; text is rebuilt from token ids.
            data = json.loads(Path(jpath).read_bytes().decode("utf-8", errors="replace"))
            os.remove(jpath)
            # Group across segments: a segment can end in the middle of a multi-byte Urdu letter.
            words = tokens_to_words([t for seg in data.get("transcription", []) for t in seg.get("tokens", [])])
            results[p] = {"text": " ".join(w["w"] for w in words), "words": words}
        if progress:
            print(f"  {min(i + batch, len(wav_paths))}/{len(wav_paths)} files", flush=True)
    return results, wall

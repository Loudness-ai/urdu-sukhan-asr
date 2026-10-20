"""Round-6 extras, built on the GPU server from data already downloaded there.

  python scripts/make_r6_extras.py stress   # data/stress/*: the pass/fail robustness test (never trained on)
  python scripts/make_r6_extras.py tests    # data/tests/*: held-out IndicVoices and UrduSpeech test sets
  python scripts/make_r6_extras.py train    # data/nospeech, data/codeswitch: extra training clips

Stress subsets and what a good model does on them:
  noise, music, silence  -> no text at all (anything it writes is invented)
  quiet, phone           -> the right words although the audio is 30 dB softer / band-limited to a phone line
  short                  -> one- or two-word replies ("جی", "ہاں")
  english, mixed         -> English in English, Urdu + English together
  long                   -> 25-30 s of continuous speech without dropping or repeating anything
MUSAN files are split by name: one in ten is kept for the stress test and never used in training.
"""

import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wlite.audio import load_audio  # noqa: E402
from wlite.text import clean_label  # noqa: E402

SR = 16000
MUSAN = ROOT / "data" / "musan" / "musan"
rng = random.Random(0)


def held_out(path):
    return int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 10 == 0


def musan_files(kind):
    """MUSAN noise, or music without singing (songs would teach the model to ignore words)."""
    if kind == "noise":
        return sorted((MUSAN / "noise").rglob("*.wav"))
    vocals = {}
    for ann in (MUSAN / "music").rglob("ANNOTATIONS"):
        for line in ann.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                vocals[parts[0]] = parts[2].upper() == "Y"
    return sorted(p for p in (MUSAN / "music").rglob("*.wav") if not vocals.get(p.stem, True))


def segment(path, seconds):
    info = sf.info(path)
    n = int(seconds * info.samplerate)
    start = rng.randrange(0, max(1, info.frames - n))
    x, sr = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR)
    return x


def read(manifest):
    manifest = Path(manifest)
    rows = [json.loads(l) for l in manifest.read_text("utf-8").splitlines() if l.strip()]
    for r in rows:
        r["audio_path"] = str((manifest.parent / r["audio"]).resolve())
    return rows


class Writer:
    def __init__(self, folder, name):
        self.dir = ROOT / "data" / folder
        (self.dir / name).mkdir(parents=True, exist_ok=True)
        self.name, self.f, self.n = name, open(self.dir / f"{name}.jsonl", "w", encoding="utf-8"), 0

    def add(self, audio, text, language="ur", **extra):
        rel = f"{self.name}/{self.n}.flac"
        sf.write(self.dir / rel, np.clip(audio, -1, 1).astype(np.float32), SR)
        self.f.write(json.dumps({"id": f"{self.name}-{self.n}", "audio": rel, "text": text, "language": language,
                                 "duration": round(len(audio) / SR, 2), **extra}, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self):
        self.f.close()
        print(f"{self.dir.name}/{self.name}: {self.n} clips")


def scaled(audio, db):
    return audio * 10 ** (db / 20)


def phone(audio):
    import soxr
    x = soxr.resample(soxr.resample(audio, SR, 8000), 8000, SR)[: len(audio)]
    return np.clip(x * 1.5, -0.5, 0.5)


def iv_valid_rows():
    """IndicVoices' own validation file: audio, text, speaker. Never used for training."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local = hf_hub_download("ai4bharat/IndicVoices", "urdu/valid-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(local, columns=["audio_filepath", "text", "speaker_id", "duration"]).to_pylist()


def decode(raw):
    import soxr
    x, sr = sf.read(io.BytesIO(raw), dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    return soxr.resample(x, sr, SR) if sr != SR else x


def stress():
    speech = read(ROOT / "data/cv20k/test_unique.jsonl") + read(ROOT / "data/proofread/novel/test_book2.jsonl")
    rng.shuffle(speech)
    w = Writer("stress", "noise")
    for p in rng.sample([p for p in musan_files("noise") if held_out(p)], 40):
        w.add(segment(p, rng.uniform(3, 15)), "", "auto")
    w.close()
    w = Writer("stress", "music")
    for p in rng.sample([p for p in musan_files("music") if held_out(p)], 30):
        w.add(segment(p, rng.uniform(3, 15)), "", "auto")
    w.close()
    w = Writer("stress", "silence")
    for k in range(30):  # digital silence, then faint room hiss / hum
        n = int(rng.uniform(2, 15) * SR)
        x = np.zeros(n, np.float32) if k < 10 else np.random.default_rng(k).standard_normal(n).astype(np.float32) * 10 ** (rng.uniform(-70, -50) / 20)
        if k >= 20:
            x += 10 ** (-55 / 20) * np.sin(2 * np.pi * 50 * np.arange(n) / SR).astype(np.float32)
        w.add(x, "", "auto")
    w.close()
    w = Writer("stress", "quiet")
    for r in speech[:40]:
        w.add(scaled(load_audio(r["audio_path"]), -30), r["text"])
    w.close()
    w = Writer("stress", "phone")
    for r in speech[40:80]:
        w.add(phone(load_audio(r["audio_path"])), r["text"])
    w.close()
    iv = iv_valid_rows()
    w = Writer("stress", "short")
    for r in rng.sample([r for r in iv if r["duration"] and float(r["duration"]) < 1.5 and r["text"].strip()], 40):
        w.add(decode(r["audio_filepath"]["bytes"]), clean_label(r["text"], "ur"))
    w.close()
    en = read(ROOT / "data/fleurs/en_us/test.jsonl")
    rng.shuffle(en)
    w = Writer("stress", "english")
    for r in en[:30]:
        w.add(load_audio(r["audio_path"]), r["text"], "auto:en")
    w.close()
    w = Writer("stress", "mixed")
    for a, b in zip(speech[80:100], en[30:50]):
        gap = np.zeros(int(0.4 * SR), np.float32)
        x = np.concatenate([load_audio(a["audio_path"]), gap, load_audio(b["audio_path"])])[: 30 * SR]
        w.add(x, f'{a["text"]} {b["text"]}', "auto")
    w.close()
    w = Writer("stress", "long")
    pool = speech[100:]
    for _ in range(15):
        parts, total = [], 0.0
        while total < 25 and pool:
            r = pool.pop()
            if total + r["duration"] > 29.5:
                break
            parts.append(r); total += r["duration"] + 0.3
        if len(parts) >= 2:
            gap = np.zeros(int(0.3 * SR), np.float32)
            x = np.concatenate(sum([[load_audio(r["audio_path"]), gap] for r in parts], []))
            w.add(x, " ".join(r["text"] for r in parts))
    w.close()


def tests():
    iv = iv_valid_rows()
    rng.shuffle(iv)
    per, w = {}, Writer("tests", "indicvoices")
    for r in iv:  # up to 3 clips per speaker, 300 in all
        if per.get(r["speaker_id"], 0) >= 3 or not r["text"].strip() or w.n >= 300:
            continue
        per[r["speaker_id"]] = per.get(r["speaker_id"], 0) + 1
        w.add(decode(r["audio_filepath"]["bytes"]), clean_label(r["text"], "ur"), speaker=r["speaker_id"])
    w.close()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    local = hf_hub_download("humairawan/UrduSpeech", "Urdu/test-00000-of-00007.parquet", repo_type="dataset")
    rows = pq.read_table(local, columns=["audio", "text"]).to_pylist()
    w = Writer("tests", "urduspeech")
    for r in rng.sample(rows, 100):
        w.add(decode(r["audio"]["bytes"]), clean_label(r["text"], "ur"))
    w.close()


def train():
    files = [p for p in musan_files("noise") + musan_files("music") if not held_out(p)]
    w = Writer("nospeech", "train")
    for _ in range(2500):  # background with nothing said: the answer is no text
        w.add(segment(rng.choice(files), rng.uniform(1.5, 25)), "")
    for k in range(300):  # silence, faint hiss and mains hum, sometimes very short
        n = int(rng.uniform(0.5, 20) * SR)
        x = np.random.default_rng(1000 + k).standard_normal(n).astype(np.float32) * 10 ** (rng.uniform(-80, -45) / 20)
        if k % 3 == 0:
            x += 10 ** (rng.uniform(-60, -40) / 20) * np.sin(2 * np.pi * rng.choice([50, 60]) * np.arange(n) / SR).astype(np.float32)
        w.add(x, "")
    w.close()
    ur = read(ROOT / "data/fleurs/ur_pk/train.jsonl") + read(ROOT / "data/cv20k/train.jsonl")
    en = read(ROOT / "data/fleurs/en_us/train.jsonl")
    w = Writer("codeswitch", "train")
    for _ in range(800):  # Urdu then English (sometimes the other way), as people switch mid-sentence
        a, b = rng.choice(ur), rng.choice(en)
        if a["duration"] + b["duration"] > 28:
            continue
        first, second = (a, b) if rng.random() < 0.7 else (b, a)
        gap = np.zeros(int(rng.uniform(0.15, 0.5) * SR), np.float32)
        x = np.concatenate([load_audio(first["audio_path"]), gap, load_audio(second["audio_path"])])
        w.add(x, f'{first["text"]} {second["text"]}')
    w.close()


if __name__ == "__main__":
    {"stress": stress, "tests": tests, "train": train}[sys.argv[1]]()

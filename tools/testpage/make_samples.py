"""Pick test clips (never trained on) for the CPU test page: tools/testpage/samples/ + samples.json.

  python tools/testpage/make_samples.py

Clips are converted to 16 kHz mono WAV, as the app feeds whisper.cpp.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wlite.audio import load_audio  # noqa: E402

SETS = [  # (name shown, manifest, clips, language for scoring)
    ("Book 2 (test)", "data/proofread/novel/test_book2.jsonl", 10, "ur"),
    ("Book 1 (test)", "data/proofread/novel/test_book1.jsonl", 8, "ur"),
    ("Common Voice, new speakers", "data/cv20k/test_unique.jsonl", 10, "ur"),
    ("IndicVoices", "data/tests/indicvoices.jsonl", 10, "ur"),
    ("UrduSpeech", "data/tests/urduspeech.jsonl", 6, "ur"),
    ("FLEURS Urdu", "data/fleurs/ur_pk/test.jsonl", 8, "ur"),
    ("English (FLEURS)", "data/stress/english.jsonl", 5, "en"),
    ("Phone line", "data/stress/phone.jsonl", 5, "ur"),
    ("Short replies", "data/stress/short.jsonl", 6, "ur"),
    ("Silence and hiss (no speech)", "data/stress/silence.jsonl", 5, "ur"),
    ("Noise (no speech)", "data/stress/noise.jsonl", 5, "ur"),
    ("Music (no speech)", "data/stress/music.jsonl", 4, "ur"),
]


def main():
    out = ROOT / "tools" / "testpage" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    items = []
    for name, manifest, n, lang in SETS:
        path = ROOT / manifest
        rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
        for k, r in enumerate(rng.sample(rows, min(n, len(rows)))):
            audio = load_audio(str((path.parent / r["audio"]).resolve()))
            fname = f"{path.parent.name}_{path.stem}_{k}.wav"
            sf.write(out / fname, np.clip(audio, -1, 1), 16000, subtype="PCM_16")
            items.append({"set": name, "file": fname, "text": r["text"], "language": lang,
                          "duration": round(len(audio) / 16000, 2)})
    (ROOT / "tools" / "testpage" / "samples.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    print(f"{len(items)} clips, {sum(i['duration'] for i in items) / 60:.1f} min")


if __name__ == "__main__":
    main()

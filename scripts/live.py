"""Live test page: speak Urdu into the microphone (or drop in an audio file) and see each phrase transcribed.

  python scripts/live.py                      # http://127.0.0.1:8770
  python scripts/live.py --model models/ggml/ggml-novel-r3-q4_0.bin --port 8771

Runs whisper.cpp's own server, which serves live/index.html plus /inference and /load, so the page can
switch between models. It uses the Mac's GPU: stop it before training or scoring models.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGML = ROOT / "models" / "ggml"
LIVE = ROOT / "live"
VAD = ROOT / "models" / "vad" / "ggml-silero-v5.1.2.bin"
# (label shown in the page, file); missing files are left out.
MODELS = [
    ("Best: round 6 · 874 MB", "ggml-r6c2775-q8_0.bin"),
    ("Blend of rounds 4+5 (best on the audiobook) · 874 MB", "ggml-blend-r4r5-q8_0.bin"),
    ("Round 5 · 874 MB", "ggml-novel-r5-q8_0.bin"),
    ("Round 4 · 874 MB", "ggml-novel-r4-q8_0.bin"),
    ("Round 3 · 874 MB", "ggml-novel-r3-q8_0.bin"),
    ("Round 3 · 574 MB", "ggml-novel-r3-q5_0.bin"),
    ("Round 3 · 474 MB", "ggml-novel-r3-q4_0.bin"),
    ("Round 2 · 874 MB", "ggml-novel-r2-q8_0.bin"),
    ("Round 1 · 874 MB", "ggml-novel-r1-q8_0.bin"),
    ("Stock turbo · 874 MB", "ggml-large-v3-turbo-q8_0.bin"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(GGML / MODELS[0][1]))
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--lang", default="auto", help="'auto' detects Urdu/English per phrase (the page redoes other languages as Urdu)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--no-vad", action="store_true", help="send every phrase to the model, even with no speech in it")
    args = ap.parse_args()

    start = str(Path(args.model).resolve())
    listed = [{"label": label, "path": str(GGML / f), "loaded": str(GGML / f) == start}
              for label, f in MODELS if (GGML / f).exists()]
    (LIVE / "models.json").write_text(json.dumps(listed, ensure_ascii=False, indent=1))

    exe = shutil.which("whisper-server")
    if not exe:
        raise SystemExit("whisper-server not found (brew install whisper-cpp)")
    # -nt: decode without timestamp tokens, the way the fine-tunes are scored.
    # -mc 0: keep no text from one phrase to the next (an English phrase used to derail the next Urdu one).
    # Temperature retries stay on: in testing they removed every runaway output on unheard voices.
    # The page also checks for repeats and retries in the other language.
    cmd = [exe, "-m", start, "-l", args.lang, "-bs", str(args.beam_size), "-nt", "-mc", "0",
           "--host", "127.0.0.1", "--port", str(args.port), "--public", str(LIVE)]
    # Voice detector (Silero): audio with no speech in it never reaches the model, which removed nearly all
    # invented text on noise and silence in testing. Split only at 2 s pauses (the page already cuts phrases)
    # and keep 300 ms around speech so soft word endings survive.
    if VAD.exists() and not args.no_vad:
        cmd += ["--vad", "-vm", str(VAD), "-vt", "0.5", "-vsd", "2000", "-vp", "300"]
    print(f"live test on http://127.0.0.1:{args.port}", flush=True)
    os.execv(exe, cmd)


if __name__ == "__main__":
    main()

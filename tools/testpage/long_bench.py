"""Long-audio speed and accuracy on this machine's CPU: whisper.cpp (the app engine) vs faster-whisper (CTranslate2).

  fw-venv/bin/python tools/testpage/long_bench.py long_test.wav long_test.ref.txt --threads 8

Every engine gets the same file, language Urdu, beam 5, the Silero voice detector to cut it at pauses, and no text
carried between pieces. Prints wall time, times faster than real time, and words/letters wrong against the
reference. Results are appended to long_bench.jsonl.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from textnorm import normalize_for_eval  # noqa: E402

CLI = ROOT / "whisper.cpp/build/bin/whisper-cli"
VAD = ROOT / "models/vad/ggml-silero-v5.1.2.bin"


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(ref, hyp):
    r, h = normalize_for_eval(ref, "ur"), normalize_for_eval(hyp, "ur")
    wer = 100 * edits(r.split(), h.split()) / len(r.split())
    cer = 100 * edits(r.replace(" ", ""), h.replace(" ", "")) / len(r.replace(" ", ""))
    return round(wer, 1), round(cer, 1)


def whisper_cpp(model, wav, threads):
    out = Path("/tmp/long_bench_cpp")
    cmd = ["nice", "-n", "10", str(CLI), "-m", str(model), "-l", "ur", "-bs", "5", "-nt", "-mc", "0", "-t", str(threads),
           "--vad", "-vm", str(VAD), "-vt", "0.5", "-vsd", "2000", "-vp", "300", "-otxt", "-of", str(out), "-f", str(wav)]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True)
    wall = time.perf_counter() - t0
    return " ".join(Path(str(out) + ".txt").read_text("utf-8", errors="replace").split()), wall


def faster_whisper(model_dir, wav, threads, batched, beam):
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads)
    load = time.perf_counter() - t0
    import soundfile as sf

    t0 = time.perf_counter()
    audio, sr = sf.read(str(wav), dtype="float32")  # 16 kHz mono already; decoding is timed with the rest
    assert sr == 16000
    if batched:
        pipe = BatchedInferencePipeline(model=model)
        segs, _ = pipe.transcribe(audio, language="ur", beam_size=beam, batch_size=batched, without_timestamps=True)
    else:
        segs, _ = model.transcribe(audio, language="ur", beam_size=beam, vad_filter=True,
                                   condition_on_previous_text=False, without_timestamps=True)
    text = " ".join(s.text.strip() for s in segs)  # segments are generated lazily: this runs the model
    return " ".join(text.split()), time.perf_counter() - t0, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("ref")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    import soundfile as sf

    dur = sf.info(args.wav).duration
    ref = Path(args.ref).read_text("utf-8")
    runs = [
        ("whisper.cpp teacher q8_0", lambda: whisper_cpp(ROOT / "models/ggml/ggml-r6c2775-q8_0.bin", args.wav, args.threads) + (0,)),
        ("whisper.cpp teacher q4_0", lambda: whisper_cpp(ROOT / "models/ggml/ggml-r6c2775-q4_0.bin", args.wav, args.threads) + (0,)),
        ("faster-whisper teacher int8", lambda: faster_whisper(ROOT / "models/ct2/r6c2775-int8", args.wav, args.threads, 0, 5)),
        ("faster-whisper teacher int8, batched 8", lambda: faster_whisper(ROOT / "models/ct2/r6c2775-int8", args.wav, args.threads, 8, 5)),
        ("faster-whisper teacher int8, batched 8, greedy", lambda: faster_whisper(ROOT / "models/ct2/r6c2775-int8", args.wav, args.threads, 8, 1)),
    ]
    print(f"audio {dur / 60:.1f} min, {args.threads} threads", flush=True)
    for name, fn in runs:
        if args.only and name not in args.only:
            continue
        text, wall, load = fn()
        wer, cer = score(ref, text)
        row = {"engine": name, "threads": args.threads, "audio_min": round(dur / 60, 1), "wall_min": round(wall / 60, 2),
               "x_realtime": round(dur / wall, 2), "load_s": round(load, 1), "wer": wer, "cer": cer, "words": len(text.split())}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open(ROOT / "long_bench.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(ROOT / f"long_bench_{name.replace(' ', '_').replace(',', '')}.txt").write_text(text, "utf-8")


if __name__ == "__main__":
    main()

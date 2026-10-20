"""Transcribe an audio file, short or hours long, with Urdu Sukhan ASR v1 in faster-whisper (CTranslate2).

    python examples/transcribe_faster_whisper.py recording.wav
    python examples/transcribe_faster_whisper.py lecture.flac --batch-size 8 --timestamps

Downloads only the `ctranslate2/` folder of the model the first time. The Silero voice detector cuts the audio into
speech pieces, which are transcribed in batches; this is the fastest way to transcribe long recordings on a CPU.
Uses int8 on CPUs and float16 on NVIDIA GPUs.
"""
import argparse
import time

import soundfile as sf
import soxr
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub import snapshot_download

SAMPLE_RATE = 16000


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE)
    return audio


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio")
    ap.add_argument("--model", default="Loudness-ai/urdu-sukhan-asr-v1",
                    help="Hugging Face id, or a local folder that holds the CTranslate2 model files")
    ap.add_argument("--language", default="ur", help="ur, en, or none to detect it")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8, help="speech pieces decoded together")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads; leave some cores free on a shared server")
    ap.add_argument("--device", default="auto", help="cpu, cuda or auto")
    ap.add_argument("--timestamps", action="store_true", help="print start and end time of each piece")
    args = ap.parse_args()

    path = args.model
    if "/" in path and not path.startswith((".", "/")):
        path = snapshot_download(path, allow_patterns=["ctranslate2/*"]) + "/ctranslate2"
    device = args.device
    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = WhisperModel(path, device=device, compute_type="float16" if device == "cuda" else "int8",
                         cpu_threads=args.threads)
    pipe = BatchedInferencePipeline(model)

    audio = load_audio(args.audio)
    start = time.time()
    segments, _ = pipe.transcribe(audio, language=None if args.language == "none" else args.language,
                                  beam_size=args.beams, batch_size=args.batch_size, vad_filter=True,
                                  without_timestamps=not args.timestamps, condition_on_previous_text=False)
    for seg in segments:  # transcription happens while iterating
        if args.timestamps:
            print(f"[{seg.start:7.1f} - {seg.end:7.1f}] {seg.text.strip()}", flush=True)
        else:
            print(seg.text.strip(), flush=True)
    took = time.time() - start
    seconds = len(audio) / SAMPLE_RATE
    print(f"\n[{seconds:.1f} s of audio in {took:.1f} s on {device}: {seconds / took:.2f} times real time]")


if __name__ == "__main__":
    main()

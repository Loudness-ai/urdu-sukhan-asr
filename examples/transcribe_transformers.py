"""Transcribe an audio file with Urdu Sukhan ASR v1 in Hugging Face Transformers.

    python examples/transcribe_transformers.py speech.wav
    python examples/transcribe_transformers.py interview.flac --beams 5 --language ur

Runs on an NVIDIA GPU (fp16), an Apple GPU or the CPU (fp32). Recordings longer than 30 seconds are cut into
30-second chunks and transcribed in batches. Reads WAV, FLAC, OGG and MP3; convert other formats first, for example
`ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav`.
"""
import argparse
import time

import soundfile as sf
import soxr
import torch
from transformers import pipeline

SAMPLE_RATE = 16000


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # mono
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE)
    return audio


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio")
    ap.add_argument("--model", default="Loudness-ai/urdu-sukhan-asr-v1", help="Hugging Face id or local folder")
    ap.add_argument("--language", default="ur", help="ur, en, or none to detect it")
    ap.add_argument("--beams", type=int, default=1, help="1 = greedy (fast); 5 = beam search (slightly better)")
    ap.add_argument("--batch-size", type=int, default=4, help="30-second chunks decoded together")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32  # fp16 on Apple GPUs can loop on near-silent audio
    else:
        device, dtype = "cpu", torch.float32

    asr = pipeline("automatic-speech-recognition", model=args.model, dtype=dtype, device=device)
    audio = load_audio(args.audio)
    generate_kwargs = {"task": "transcribe", "num_beams": args.beams}
    if args.language != "none":
        generate_kwargs["language"] = args.language

    start = time.time()
    result = asr({"raw": audio, "sampling_rate": SAMPLE_RATE}, chunk_length_s=30, batch_size=args.batch_size,
                 ignore_warning=True, generate_kwargs=generate_kwargs)
    took = time.time() - start
    print(result["text"].strip())
    print(f"\n[{len(audio) / SAMPLE_RATE:.1f} s of audio in {took:.1f} s on {device}]", flush=True)


if __name__ == "__main__":
    main()

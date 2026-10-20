"""Audio loading, resampling and silence-based chunking. Everything is 16 kHz mono float32."""

import shutil
import subprocess

import numpy as np
import soundfile as sf

SR = 16000


def load_audio(path):
    """Decode any audio/video file to 16 kHz mono float32."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        return _load_with_ffmpeg(path)
    if sr != SR:
        import soxr

        audio = soxr.resample(audio, sr, SR, quality="HQ")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _load_with_ffmpeg(path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"Cannot decode {path}: install ffmpeg (brew install ffmpeg) for this format.")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def write_wav(path, audio):
    sf.write(str(path), audio, SR, subtype="PCM_16")


def speech_chunks(audio, max_s=25.0, target_s=15.0, min_s=1.0):
    """Split long audio into clips of speech, cutting only in pauses.

    Uses the Silero VAD bundled with faster-whisper. Neighbouring speech regions
    are merged until a clip reaches ~target_s; no clip exceeds max_s, which keeps
    every clip inside Whisper's 30 s training window.
    Returns a list of (start_sample, end_sample).
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=250,
        max_speech_duration_s=max_s,
        min_silence_duration_ms=300,
        speech_pad_ms=150,
    )
    regions = get_speech_timestamps(audio, vad_options=opts, sampling_rate=SR)
    chunks = []
    cur = None
    for r in regions:
        s, e = r["start"], r["end"]
        if cur is None:
            cur = [s, e]
        elif (e - cur[0]) / SR <= max_s and (cur[1] - cur[0]) / SR < target_s:
            cur[1] = e
        else:
            chunks.append(tuple(cur))
            cur = [s, e]
    if cur:
        chunks.append(tuple(cur))
    return [(s, e) for s, e in chunks if (e - s) / SR >= min_s]

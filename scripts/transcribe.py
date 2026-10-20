"""Turn long recordings into short clips with draft transcripts, ready to proofread.

  python scripts/transcribe.py ~/Desktop/urdu_audio/*.mp3 --project batch1
  python scripts/proofread.py batch1

Each recording is cut at pauses into clips of up to ~25 s (Whisper's training
window is 30 s), and every clip gets a draft from whisper.cpp with per-word
confidence. Running it again with more files adds them to the same project;
clips already drafted are skipped, so it is safe to rerun after an interruption.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.audio import SR, load_audio, speech_chunks  # noqa: E402
from wlite.project import Project  # noqa: E402
from wlite.text import clean_label  # noqa: E402
from wlite.whispercpp import transcribe  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".mp4", ".mov", ".mkv", ".webm"}


def expand(inputs):
    for p in map(Path, inputs):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.exists():
            yield p
        else:
            print(f"skip (not found): {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="audio/video files or folders")
    ap.add_argument("--project", required=True)
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--model", default="large-v3-turbo-q8_0", help="stock whisper.cpp name or path to a .bin")
    ap.add_argument("--max-clip", type=float, default=25.0, help="longest clip in seconds")
    ap.add_argument("--prompt", default=None, help="optional text to steer spelling/style")
    ap.add_argument("--batch", type=int, default=40, help="clips drafted (and saved) per step")
    args = ap.parse_args()

    proj = Project(args.project, create=True)
    import soundfile as sf

    total_s = total_wall = 0.0
    for src in expand(args.inputs):
        src = src.resolve()
        print(f"cutting {src.name} ...", flush=True)
        audio = load_audio(src)
        stem = re.sub(r"[^\w\-]+", "_", src.stem)[:60]
        chunks = speech_chunks(audio, max_s=args.max_clip)
        proj.reload()
        # Cutting is deterministic, so a rerun after an interruption skips clips already drafted.
        todo = [(k, s, e) for k, (s, e) in enumerate(chunks) if f"{stem}_{k:04d}" not in proj.by_id]
        print(f"  {len(chunks)} clips from {len(audio) / SR / 60:.1f} min; {len(todo)} to draft", flush=True)
        for i in range(0, len(todo), args.batch):
            batch = []
            for k, s, e in todo[i : i + args.batch]:
                clip = proj.clip_dir / f"{stem}_{k:04d}.flac"
                sf.write(clip, audio[s:e], SR)
                batch.append({
                    "id": f"{stem}_{k:04d}",
                    "source": str(src),
                    "clip": clip.name,
                    "start": round(s / SR, 2),
                    "end": round(e / SR, 2),
                    "duration": round((e - s) / SR, 2),
                    "language": args.lang,
                    "status": "todo",
                    "draft": "",
                    "words": [],
                    "text": None,
                    "model": args.model,
                })
            paths = [str(proj.clip_dir / it["clip"]) for it in batch]
            res, wall = transcribe(args.model, paths, args.lang, prompt=args.prompt, progress=False)
            for it, p in zip(batch, paths):
                it["draft"] = clean_label(res[p]["text"], args.lang)
                it["words"] = [{**w, "w": clean_label(w["w"], args.lang)} for w in res[p]["words"]]
            # Saved per batch, so the proofreading page can show clips while the rest are drafted.
            proj.update(lambda items: items.extend(batch))
            total_s += sum(it["duration"] for it in batch)
            total_wall += wall
            print(f"  drafted {min(i + args.batch, len(todo))}/{len(todo)} "
                  f"({total_s / total_wall:.1f}x real time)", flush=True)

    if total_wall:
        print(f"Done: {total_s / 60:.1f} min of speech in {total_wall / 60:.1f} min.")
    print(f"Proofread with: python scripts/proofread.py {args.project}")


if __name__ == "__main__":
    main()

"""Title-card videos of the example clips, for GitHub (which plays uploaded videos but not audio)."""
import html, json, subprocess, sys, urllib.request
from pathlib import Path

REL = Path(sys.argv[1]); OUT = Path(sys.argv[2])
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT = (REL / "paper/fonts/NotoNastaliqUrdu-Regular.ttf").as_uri()
EX = "https://huggingface.co/datasets/Loudness-ai/urdu-sukhan-asr-examples/resolve/main/examples/{}.mp3"
sys.path.insert(0, str(REL / "paper/data/examples"))
d = json.loads((REL / "paper/data/examples/examples.json").read_text(encoding="utf-8"))
src = (REL / "paper/data/examples/make_pages_section.py").read_text(encoding="utf-8")
ns = {}
exec(src[src.index("NAMES = ["):src.index("assert len(")], ns)  # NAMES, KIND, MEANING
clips = [(n, ns["KIND"][e["cat"]], e["ref"]) for n, e in zip(ns["NAMES"], d["examples"])]
clips += [("10-faint-hiss", "No speech: faint hiss and hum", ""), ("11-noise", "No speech: noise (MUSAN)", "")]
for k, (name, kind, ref) in enumerate(clips, 1):
    mp3 = OUT / f"{name}.mp3"
    if not mp3.exists():
        urllib.request.urlretrieve(EX.format(name), mp3)
    body = (f'<div class="ref" dir="auto">{html.escape(ref)}</div>' if ref else '<div class="none">(no speech)</div>')
    (OUT / f"{name}.html").write_text(f"""<!doctype html><meta charset="utf-8"><style>
@font-face {{ font-family: N; src: url('{FONT}'); }}
html, body {{ margin: 0; width: 1280px; height: 360px; background: #0f172a; color: #e2e8f0; font-family: -apple-system, Helvetica, Arial, sans-serif; }}
.top {{ position: absolute; left: 48px; top: 36px; font-size: 22px; color: #94a3b8; }}
.kind {{ position: absolute; left: 48px; top: 72px; font-size: 34px; font-weight: 600; color: #f8fafc; }}
.ref {{ position: absolute; left: 48px; right: 48px; top: 150px; font-family: N, sans-serif; font-size: 34px; line-height: 64px;
        text-align: right; unicode-bidi: plaintext; max-height: 192px; overflow: hidden; }}
.none {{ position: absolute; left: 48px; top: 170px; font-size: 30px; color: #94a3b8; font-style: italic; }}
</style><div class="top">Urdu Sukhan ASR v1 · example {k} of {len(clips)} · press play to listen</div>
<div class="kind">{html.escape(kind)}</div>{body}""", encoding="utf-8")
    png = OUT / f"{name}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    "--virtual-time-budget=4000", "--window-size=1280,360", f"--screenshot={png}",
                    (OUT / f"{name}.html").as_uri()], check=True, capture_output=True)
    secs = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
                          capture_output=True, text=True, check=True).stdout.strip()
    # -t, not just -shortest: with a looped still image the video track otherwise runs on past the audio
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "2", "-i", str(png), "-i", str(mp3), "-t", secs,
                    "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "2", "-c:a", "aac",
                    "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(OUT / f"{name}.mp4")], check=True)
    print(name, (OUT / f"{name}.mp4").stat().st_size // 1024, "KB")

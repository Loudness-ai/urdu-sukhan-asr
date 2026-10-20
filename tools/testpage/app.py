"""CPU test page for the Urdu student and teacher models (whisper.cpp on this server's CPU, no GPU).

  python3 tools/testpage/app.py --port 8090 --token <secret>

Open http://<server>:8090/<secret>/ , or https://<server>:8443/<secret>/ (self-signed certificate) to record live:
browsers only allow the microphone on https pages. Upload any audio (the browser converts it to 16 kHz WAV) or run the built-in
test clips (never trained on, with human transcripts). Each chosen model transcribes it the way the app does
(beam 5, temperature fallback, no timestamps, language auto-detect, no carried context); the page shows the text,
the processing time (model loading excluded and shown apart), times faster than real time, and for test clips the
words and letters wrong. One job runs at a time; uploads are deleted as soon as they are transcribed.
"""

import argparse
import json
import os
import queue
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = Path(os.environ.get("WHISPER_CLI", Path.home() / "src/whisper.cpp/build/bin/whisper-cli"))
NICE = int(os.environ.get("WHISPER_NICE", "0"))  # on a shared server, let other services go first
VAD_MODEL = ROOT / "models/vad/ggml-silero-v5.1.2.bin"
MODELS = [  # id, label, file (models missing on this server are left out)
    ("teacher-q8", "Teacher (round 6) · q8_0", ROOT / "models/ggml/ggml-r6c2775-q8_0.bin"),
    ("teacher-q4", "Teacher (round 6) · q4_0", ROOT / "models/ggml/ggml-r6c2775-q4_0.bin"),
    ("s1-q8", "Student S1 · q8_0", ROOT / "models/ggml/ggml-student-s1-q8_0.bin"),
    ("s1-q5", "Student S1 · q5_0", ROOT / "models/ggml/ggml-student-s1-q5_0.bin"),
    ("s1-q4", "Student S1 · q4_0", ROOT / "models/ggml/ggml-student-s1-q4_0.bin"),
]
LANGS = {"auto": "auto", "ur": "ur", "en": "en"}
MAX_UPLOAD = 60 * 1024 * 1024  # 16 kHz 16-bit mono WAV: about 30 minutes
MARKS = re.compile(r"[ً-ٰٟ]")

import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from textnorm import normalize_for_eval  # noqa: E402

SAMPLES = json.loads((HERE / "samples.json").read_text("utf-8"))
JOBS, ORDER, WORK = {}, [], queue.Queue()


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def repeats(words, i, n):
    k = 1
    while words[i + k * n: i + (k + 1) * n] == words[i: i + n]:
        k += 1
    return k


def looping(hyp, ref):
    """As eval_hf.runaway: more than twice the words said, or a 1-6 word phrase 4+ times in a row more often than
    the speaker said it."""
    w, r = MARKS.sub("", hyp).split(), MARKS.sub("", ref).split()
    if len(w) > 2 * max(3, len(r)):
        return True
    for n in range(1, 7):
        for i in range(0, len(w) - 4 * n + 1):
            k = repeats(w, i, n)
            if k >= 4:
                said = max([repeats(r, j, n) for j in range(len(r) - n + 1) if r[j: j + n] == w[i: i + n]] or [0])
                if k > said + 1:
                    return True
    return False


def run_model(model_file, wavs, threads, vad, lang="auto", prompt=""):
    """One whisper-cli process for all files (the model loads once, as in the app). Returns texts, seconds, load s.
    lang "auto" costs an extra encoder pass per clip (whisper.cpp encodes once to detect the language)."""
    cmd = ([ "nice", "-n", str(NICE)] if NICE else []) + [
        str(CLI), "-m", str(model_file), "-l", lang, "-bs", "5", "-nt", "-mc", "0", "-t", str(threads), "-otxt"]
    if vad:
        cmd += ["--vad", "-vm", str(VAD_MODEL), "-vt", "0.5", "-vsd", "2000", "-vp", "300"]
    if prompt:  # names and words to prefer, given to the decoder as earlier text
        cmd += ["--prompt", prompt]
    for w in wavs:
        cmd += ["-f", str(w)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    wall = time.perf_counter() - t0
    load = 0.0
    m = re.search(r"load time =\s*([\d.]+) ms", proc.stderr)
    if m:
        load = float(m.group(1)) / 1000
    texts = []
    for w in wavs:
        txt = Path(str(w) + ".txt")
        texts.append(" ".join(txt.read_text("utf-8", errors="replace").split()) if txt.exists() else "")
        txt.unlink(missing_ok=True)
    if proc.returncode != 0 and not any(texts):
        raise RuntimeError(proc.stderr[-600:])
    return texts, max(0.0, wall - load), load


def worker():
    while True:
        job = WORK.get()
        job["state"], job["started"] = "running", time.time()
        tmp = Path(tempfile.mkdtemp(prefix="testpage-"))
        try:
            wavs = []
            for k, it in enumerate(job["items"]):
                dst = tmp / f"{k}.wav"
                if it.get("upload"):
                    shutil.move(it.pop("upload"), dst)
                else:
                    os.symlink(HERE / "samples" / it["file"], dst)
                wavs.append(dst)
            for n, mid in enumerate(job["models"]):
                job["progress"] = f"model {n + 1} of {len(job['models'])}"
                label, file = next((lab, f) for i, lab, f in MODELS if i == mid)
                texts, secs, load = run_model(file, wavs, job["threads"], job["vad"], job["lang"], job["prompt"])
                audio = sum(it["duration"] for it in job["items"])
                res = {"label": label, "size_mb": round(file.stat().st_size / 1e6), "seconds": round(secs, 1),
                       "load_seconds": round(load, 1), "audio_seconds": round(audio, 1),
                       "x_realtime": round(audio / secs, 2) if secs else None, "texts": texts}
                if job["kind"] == "samples":
                    res["sets"] = score(job["items"], texts)
                job["results"][mid] = res
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001
            job["state"], job["error"] = "error", str(e)[-600:]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            job["finished"] = time.time()
            WORK.task_done()


def score(items, texts):
    sets = {}
    for it, hyp in zip(items, texts):
        s = sets.setdefault(it["set"], {"clips": 0, "we": 0, "wn": 0, "ce": 0, "cn": 0, "any": 0, "two": 0, "loops": 0,
                                        "nospeech": not it["text"]})
        s["clips"] += 1
        if not it["text"]:
            s["any"] += len(hyp.split()) >= 1
            s["two"] += len(hyp.split()) >= 2
            s["loops"] += len(hyp.split()) > 12
            continue
        ref, h = normalize_for_eval(it["text"], it["language"]), normalize_for_eval(hyp, it["language"])
        s["we"] += edits(ref.split(), h.split())
        s["wn"] += len(ref.split())
        s["ce"] += edits(ref.replace(" ", ""), h.replace(" ", ""))
        s["cn"] += len(ref.replace(" ", ""))
        s["loops"] += looping(h, ref)
    out = {}
    for name, s in sets.items():
        if s["nospeech"]:
            out[name] = {"clips": s["clips"], "nospeech": True, "any_text": s["any"], "two_plus_words": s["two"],
                         "loops": s["loops"]}
        else:
            out[name] = {"clips": s["clips"], "nospeech": False, "wer": round(100 * s["we"] / max(1, s["wn"]), 1),
                         "cer": round(100 * s["ce"] / max(1, s["cn"]), 1), "loops": s["loops"]}
    return out


def new_job(kind, items, models, threads, vad, lang="auto", prompt=""):
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "items": items, "models": models, "threads": threads, "vad": vad, "lang": lang,
           "prompt": prompt,
           "state": "queued", "progress": "", "results": {}, "created": time.time()}
    JOBS[jid] = job
    ORDER.append(jid)
    while len(ORDER) > 30:
        JOBS.pop(ORDER.pop(0), None)
    WORK.put(job)
    return jid


def public(job):
    out = {k: v for k, v in job.items() if k != "items"}
    out["items"] = [{k: v for k, v in it.items() if k != "upload"} for it in job["items"]]
    out["queue_ahead"] = sum(1 for j in JOBS.values() if j["state"] == "queued" and j["created"] < job["created"])
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "testpage"
    timeout = 120  # seconds a connection may sit idle (uploads of a few MB finish well within this)

    def log_message(self, fmt, *args):  # quieter log: path without the token
        print(time.strftime("%H:%M:%S"), self.command, self.path.replace(self.server.token, "<t>")[:80], flush=True)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _route(self):
        url = urlparse(self.path)
        prefix = f"/{self.server.token}"
        if not (url.path == prefix or url.path.startswith(prefix + "/")):
            return None, None
        return url.path[len(prefix):] or "/", {k: v[0] for k, v in parse_qs(url.query).items()}

    def _options(self, q):
        models = [m for m in q.get("models", "").split(",") if m in {i for i, _, f in MODELS if f.exists()}]
        models = models or [next(i for i, _, f in MODELS if f.exists())]
        threads = max(1, min(os.cpu_count() or 4, int(q.get("threads", "16") or 16)))
        prompt = " ".join(q.get("prompt", "").split())[:300]
        return models, threads, q.get("vad") == "1", LANGS.get(q.get("lang", "auto"), "auto"), prompt

    def do_GET(self):
        path, q = self._route()
        if path is None:
            return self._send(404, b"not found", "text/plain")
        if path == "/":
            if not self.path.split("?")[0].endswith("/"):
                self.send_response(301)
                self.send_header("Location", f"/{self.server.token}/")
                self.end_headers()
                return
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/info":
            sets = {}
            for it in SAMPLES:
                s = sets.setdefault(it["set"], {"name": it["set"], "clips": 0, "seconds": 0.0})
                s["clips"] += 1
                s["seconds"] = round(s["seconds"] + it["duration"], 1)
            return self._send(200, {"cpu": cpu_name(), "cores": os.cpu_count(), "https_port": self.server.https_port,
                                    "models": [{"id": i, "label": lab, "size_mb": round(f.stat().st_size / 1e6)}
                                               for i, lab, f in MODELS if f.exists()],
                                    "sets": list(sets.values()), "busy": WORK.qsize() + any(
                                        j["state"] == "running" for j in JOBS.values())})
        if path.startswith("/api/job/"):
            job = JOBS.get(path.rsplit("/", 1)[1])
            return self._send(200, public(job)) if job else self._send(404, {"error": "no such job"})
        if path.startswith("/sample/"):
            name = path.rsplit("/", 1)[1]
            f = HERE / "samples" / name
            if any(it["file"] == name for it in SAMPLES) and f.exists():
                return self._send(200, f.read_bytes(), "audio/wav")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path, q = self._route()
        if path is None:
            return self._send(404, b"not found", "text/plain")
        models, threads, vad, lang, prompt = self._options(q)
        if path == "/api/upload":
            n = int(self.headers.get("Content-Length") or 0)
            if not 44 < n <= MAX_UPLOAD:
                return self._send(413, {"error": "upload a WAV up to 60 MB"})
            data = self.rfile.read(n)
            if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                return self._send(400, {"error": "expected the page's 16 kHz WAV"})
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="upload-")
            os.write(fd, data)
            os.close(fd)
            duration = max(0, n - 44) / 32000  # 16 kHz, 16-bit, mono
            item = {"set": "your audio", "file": q.get("name", "upload")[:120], "text": None, "language": "ur",
                    "duration": round(duration, 2), "upload": tmp}
            return self._send(200, {"job": new_job("upload", [item], models, threads, vad, lang, prompt)})
        if path == "/api/samples":
            want = q.get("set", "all")
            items = [dict(it) for it in SAMPLES if want == "all" or it["set"] == want]
            if not items:
                return self._send(400, {"error": "no such set"})
            return self._send(200, {"job": new_job("samples", items, models, threads, vad, lang, prompt)})
        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--token", required=True)
    ap.add_argument("--https-port", type=int, default=0, help="also serve https here (needed for live recording)")
    ap.add_argument("--cert", default=str(HERE / "tls" / "cert.pem"))
    ap.add_argument("--key", default=str(HERE / "tls" / "key.pem"))
    args = ap.parse_args()
    threading.Thread(target=worker, daemon=True).start()
    servers = [ThreadingHTTPServer(("0.0.0.0", args.port), Handler)]
    if args.https_port:
        tls = ThreadingHTTPServer(("0.0.0.0", args.https_port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        # handshake in the request thread (not in accept), so a stalled client cannot block the server
        tls.socket = ctx.wrap_socket(tls.socket, server_side=True, do_handshake_on_connect=False)
        servers.append(tls)
    for srv in servers:
        srv.token, srv.https_port = args.token, args.https_port
    print(f"serving on :{args.port}" + (f" and https :{args.https_port}" if args.https_port else ""), flush=True)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()

"""Local proofreading page for a project made by scripts/transcribe.py.

  python scripts/proofread.py batch1        # then open http://localhost:8765

Every correction is written to projects/<name>/items.jsonl the moment you save it.
Only listens on this computer (127.0.0.1).
"""

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.project import Project  # noqa: E402
from wlite.corrections import apply as apply_corrections, load_rules  # noqa: E402
from wlite.text import clean_label  # noqa: E402

UI = Path(__file__).resolve().parent.parent / "proofread" / "index.html"
PUBLIC_FIELDS = ("id", "source", "clip", "start", "end", "duration", "language", "status", "draft", "words", "text")


def styled(it, rules):
    """Show drafts in the house spelling style, and pre-apply learned fixes to clips not yet proofread."""
    lang = it["language"]
    words = [{**w, "w": clean_label(w["w"], lang)} for w in it["words"] or []]
    if it["status"] == "todo" and not it.get("text") and words:
        words = apply_corrections(words, rules, lang, it.get("source") or "")
        it["draft"] = clean_label(" ".join(w["w"] for w in words), lang)
    else:
        it["draft"] = clean_label(it["draft"] or "", lang)
    it["words"] = words
    return it


_rules_cache = {}


def current_rules(proj):
    """Re-read corrections.tsv whenever it changes, so new fixes show up without a restart."""
    path = proj.dir / "corrections.tsv"
    mtime = path.stat().st_mtime if path.exists() else None
    if _rules_cache.get("mtime") != mtime:
        _rules_cache.update(mtime=mtime, rules=load_rules(path))
    return _rules_cache["rules"]


def make_handler(proj):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/project":
                proj.reload()  # picks up clips the drafting script has added since
                rules = current_rules(proj)
                items = [styled({k: it.get(k) for k in PUBLIC_FIELDS}, rules) for it in proj.items]
                return self._send(200, {"name": proj.name, "items": items, "stats": proj.stats()})
            if path == "/api/stats":
                proj.reload()
                return self._send(200, proj.stats())
            if path.startswith("/clips/"):
                return self._clip(path.removeprefix("/clips/"))
            self._send(404, {"error": "not found"})

        def _clip(self, name):
            f = (proj.clip_dir / name).resolve()
            if f.parent != proj.clip_dir.resolve() or not f.exists():
                return self._send(404, {"error": "no such clip"})
            data = f.read_bytes()
            ctype = mimetypes.guess_type(f.name)[0] or "audio/flac"
            # Range support lets the browser seek (click a word to jump to it).
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m:
                start = int(m.group(1) or 0)
                end = int(m.group(2)) if m.group(2) else len(data) - 1
                chunk = data[start : end + 1]
                return self._send(206, chunk, ctype, {"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"})
            self._send(200, data, ctype, {"Accept-Ranges": "bytes"})

        def do_POST(self):
            m = re.fullmatch(r"/api/items/([^/]+)", self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            found = []

            def apply(items):
                for it in items:
                    if m and it["id"] == m.group(1):
                        if "text" in body:
                            it["text"] = body["text"]
                        if body.get("status") in ("todo", "done", "unusable", "skipped"):
                            it["status"] = body["status"]
                        found.append(it)

            proj.update(apply)
            if not found:
                return self._send(404, {"error": "no such item"})
            self._send(200, {"ok": True, "stats": proj.stats()})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    proj = Project(args.project)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(proj))
    print(f"Proofreading '{proj.name}' ({len(proj.items)} clips): http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

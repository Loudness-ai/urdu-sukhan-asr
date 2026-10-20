"""A proofreading project: clips on disk plus items.jsonl holding draft and corrected text.

The drafting script and the proofreading server can run at the same time, so every
write goes through update(): take a file lock, re-read from disk, change, write back.
"""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"


class Project:
    def __init__(self, name, create=False):
        self.name = name
        self.dir = PROJECTS / name
        self.clip_dir = self.dir / "clips"
        self.items_path = self.dir / "items.jsonl"
        self.lock_path = self.dir / "items.lock"
        if create:
            self.clip_dir.mkdir(parents=True, exist_ok=True)
        elif not self.items_path.exists():
            raise SystemExit(f"No project '{name}' in {PROJECTS}. Create one with scripts/transcribe.py.")
        self.items = []
        self.reload()

    @contextmanager
    def _locked(self, exclusive):
        with open(self.lock_path, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _read(self):
        if self.items_path.exists():
            self.items = [json.loads(l) for l in self.items_path.read_text("utf-8").splitlines() if l.strip()]
        self.by_id = {it["id"]: it for it in self.items}

    def _write(self):
        """Atomic replace, so a crash mid-save never loses corrections."""
        tmp = self.items_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        os.replace(tmp, self.items_path)
        self.by_id = {it["id"]: it for it in self.items}

    def reload(self):
        with self._locked(exclusive=False):
            self._read()

    def update(self, change):
        """Apply change(items) to the latest on-disk state and save it."""
        with self._locked(exclusive=True):
            self._read()
            change(self.items)
            self._write()

    def save(self):
        """Overwrite the file with self.items (only for scripts that own the project)."""
        with self._locked(exclusive=True):
            self._write()

    def stats(self):
        from .text import word_char_errors

        done = [it for it in self.items if it["status"] == "done"]
        # Skipped clips are set aside, not proofread, so they don't count toward the total.
        kept = [it for it in self.items if it["status"] != "skipped"]
        out = {
            "total": len(kept),
            "done": len(done),
            "unusable": sum(it["status"] == "unusable" for it in self.items),
            "skipped": len(self.items) - len(kept),
            "total_min": round(sum(it["duration"] for it in kept) / 60, 1),
            "done_min": round(sum(it["duration"] for it in done) / 60, 1),
            "draft_wer": None,
            "draft_cer": None,
        }
        scored = [it for it in done if it.get("text")]
        if scored:
            lang = scored[0]["language"]
            wer, cer = word_char_errors([it["text"] for it in scored], [it["draft"] for it in scored], lang)
            out["draft_wer"], out["draft_cer"] = round(wer, 1), round(cer, 1)
        return out

"""Review newly finished clips: what the proofreader changed, and which fixes could be learned.

  python scripts/review_batch.py novel           # show finished clips not reviewed yet
  python scripts/review_batch.py novel --mark    # mark them reviewed
  python scripts/review_batch.py novel --wait 5  # return once 5 unreviewed finished clips exist

For each clip it prints the text, every change against the draft as the page showed it
(after house style and learned fixes), and any learned fix the proofreader undid,
which means that rule is wrong somewhere. Then it lists new repeated fixes with how
often the wrong form still appears in each book's unproofread drafts, with examples,
so a rule is only added after checking it in context.
"""

import argparse
import difflib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.corrections import apply, load_rules, suggest  # noqa: E402
from wlite.project import Project  # noqa: E402
from wlite.text import clean_label  # noqa: E402


def shown_draft(it, rules):
    """The draft exactly as the proofreading page showed it."""
    words = [{**w, "w": clean_label(w["w"], it["language"])} for w in it["words"] or []]
    return apply(words, rules, it["language"], it["source"])


def book(it):
    return Path(it["source"]).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--mark", action="store_true")
    ap.add_argument("--wait", type=int, default=0)
    args = ap.parse_args()

    proj = Project(args.project)
    state_path = proj.dir / "reviewed.json"
    reviewed = set(json.loads(state_path.read_text())) if state_path.exists() else set()

    def unreviewed():
        proj.reload()
        return [it for it in proj.items if it["status"] in ("done", "unusable") and it["id"] not in reviewed]

    if args.wait:
        while len(unreviewed()) < args.wait:
            time.sleep(10)
        print(f"{len(unreviewed())} finished clips to review")
        return

    batch_path = proj.dir / "review_batch.json"
    if args.mark:
        # Only the clips that were actually shown; anything finished since waits for the next batch.
        shown = json.loads(batch_path.read_text()) if batch_path.exists() else []
        reviewed.update(shown)
        state_path.write_text(json.dumps(sorted(reviewed)))
        batch_path.unlink(missing_ok=True)
        print(f"marked {len(shown)} clips reviewed ({len(reviewed)} total)")
        return
    new = unreviewed()
    batch_path.write_text(json.dumps([it["id"] for it in new]))

    rules = load_rules(proj.dir / "corrections.tsv")
    page_no = {it["id"]: i + 1 for i, it in enumerate(proj.items)}
    for it in new:
        print(f"\n[clip {page_no[it['id']]}] {book(it)} {'UNUSABLE' if it['status'] == 'unusable' else ''}")
        if it["status"] != "done":
            continue
        print("  text:", it["text"])
        words = shown_draft(it, rules)
        d = [t for w in words for t in w["w"].split()]
        y = clean_label(it["text"], it["language"]).split()
        changes = [f"{' '.join(d[a1:a2]) or '∅'}→{' '.join(y[b1:b2]) or '∅'}"
                   for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, d, y, autojunk=False).get_opcodes() if op != "equal"]
        print("  your changes:", " | ".join(changes) or "none")
        mine = " " + " ".join(y) + " "
        undone = [f"{w['fixed']}→{w['w']}" for w in words if "fixed" in w and f" {w['w']} " not in mine and f" {w['fixed']} " in mine]
        if undone:
            print("  !! you undid learned fixes:", " | ".join(undone))

    # New repeated fixes worth a rule: only ones that show up in this batch.
    batch_ids = {it["id"] for it in new}
    batch_pairs = set()
    for it in new:
        if it["status"] == "done":
            d = clean_label(it["draft"], it["language"]).split()
            y = clean_label(it["text"], it["language"]).split()
            for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, d, y, autojunk=False).get_opcodes():
                if op == "replace":
                    batch_pairs.add(" ".join(d[a1:a2]))
    todo = defaultdict(list)
    for it in proj.items:
        if it["status"] == "todo" and not it.get("text"):
            todo[book(it)].append(clean_label(it["draft"], it["language"]))
    print("\n== candidate rules (wrong → right; kept = times you left the wrong form as is)")
    for r in suggest(proj.items, rules):
        if r["wrong"] not in batch_pairs:
            continue
        pat = re.compile(rf"(?<!\S){re.escape(r['wrong'])}(?!\S)")
        counts = {b: sum(bool(pat.search(t)) for t in ts) for b, ts in todo.items()}
        print(f"  {r['wrong']} → {r['right']}  fixed {r['fixed']}x, kept {r['kept']}x, still in drafts: {counts}")
        shown = 0
        for b, ts in todo.items():
            for t in ts:
                m = pat.search(t)
                if m and shown < 3:
                    print(f"      {b[-12:]}: …{t[max(0, m.start() - 30):m.end() + 25]}…")
                    shown += 1
    print(f"\n{len(new)} clips in this batch (ids {sorted(batch_ids)[:1]}…)")


if __name__ == "__main__":
    main()

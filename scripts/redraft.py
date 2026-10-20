"""Re-draft the clips nobody has proofread yet with a better model.

  python scripts/redraft.py novel --model models/ggml/ggml-novel-r1-q8_0.bin

Only touches clips still "to do" with no text typed in them, re-checked at save time,
so it is safe to run while proofreading. Clips nearest to where you are come first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.project import Project  # noqa: E402
from wlite.text import clean_label  # noqa: E402
from wlite.whispercpp import transcribe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--only", default=None, help="only clips whose source path contains this text")
    args = ap.parse_args()

    proj = Project(args.project)
    model_name = Path(args.model).stem.removeprefix("ggml-")
    todo = [it for it in proj.items if it["status"] in ("todo", "skipped") and not it.get("text") and it.get("model") != model_name
            and (not args.only or args.only in it["source"])]
    print(f"{len(todo)} clips to re-draft with {model_name}", flush=True)
    done = 0
    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        paths = {it["id"]: str(proj.clip_dir / it["clip"]) for it in batch}
        res, _ = transcribe(args.model, list(paths.values()), batch[0]["language"], progress=False, no_timestamps=True)

        def apply(items):
            for it in items:
                p = paths.get(it["id"])
                if p and it["status"] in ("todo", "skipped") and not it.get("text"):  # still untouched
                    lang = it["language"]
                    it["draft"] = clean_label(res[p]["text"], lang)
                    it["words"] = [{**w, "w": clean_label(w["w"], lang)} for w in res[p]["words"]]
                    it["model"] = model_name

        proj.update(apply)
        done += len(batch)
        print(f"  {done}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()

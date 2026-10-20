"""List fixes the proofreader keeps making, to grow projects/<name>/corrections.tsv.

  python scripts/learn_corrections.py novel

Prints substitutions from finished clips that are not in corrections.tsv yet.
"kept" > 0 means the proofreader also left that word unchanged somewhere, so it
depends on context and should NOT be added. Copy the safe lines into the file;
the proofreading page picks them up on its next refresh.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.corrections import load_rules, suggest  # noqa: E402
from wlite.project import Project  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    args = ap.parse_args()
    proj = Project(args.project)
    rows = suggest(proj.items, load_rules(proj.dir / "corrections.tsv"))
    print("fixed  kept  todo-clips  wrong -> right")
    for r in rows:
        flag = "  (context-dependent)" if r["kept"] else ""
        print(f"{r['fixed']:5d} {r['kept']:5d} {r['todo_clips']:10d}  {r['wrong']}\t{r['right']}{flag}")


if __name__ == "__main__":
    main()

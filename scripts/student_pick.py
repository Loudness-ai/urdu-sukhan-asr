"""Rank a student run's checkpoints by the stress test (Hugging Face, GPU), so the pick is not just the last step.

  python scripts/student_pick.py --run models/runs/student-e16d4 --teacher models/runs/novel-r6/ckpt-2775

Runs scripts/eval_hf.py on all 9 stress sets plus the run's held-out dev clips for every ckpt-* folder not
scored yet, then ranks them: no loops and no more invented text on noise/music/silence than the teacher
first, then the lowest mean word error over the speech sets. Writes results/<run>_pick.md.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRESS = ["noise", "music", "silence", "quiet", "phone", "short", "english", "mixed", "long"]
CELL = re.compile(r"^(\S+)\s+(\S+): WER (\S+)%  CER (\S+)%  loops (\d+)/\d+(?:  invented text on (\d+)/(\d+))?")


def dev_manifest(run):
    """The run's held-out dev clips (from training manifests) as one manifest."""
    ids = json.loads((run / "dev_ids.json").read_text())
    by_path = {}
    for name in ids:
        src = ROOT / "data" / "r6" / (name.split("/", 1)[1] + ".jsonl")
        for line in src.read_text("utf-8").splitlines():
            r = json.loads(line)
            by_path[str((src.parent / r["audio"]).resolve())] = r
    out = run / "dev.jsonl"
    rows = [by_path[p] for name, paths in ids.items() if "fleurs_en" not in name for p in paths if p in by_path]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-invented", type=int, default=13, help="teacher's invented-text count on noise+music+silence")
    ap.add_argument("--only", nargs="*", help="checkpoint folder names to score (default: all)")
    args = ap.parse_args()

    run = Path(args.run)
    scores_path = run / "pick_scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    cks = sorted((p for p in run.glob("ckpt-*") if (p / "config.json").exists()), key=lambda p: int(p.name.split("-")[1]))
    todo = [p for p in cks if p.name not in scores and (not args.only or p.name in args.only)]
    manifests = [f"data/stress/{s}.jsonl" for s in STRESS] + [str(dev_manifest(run))]
    for ck in todo:
        name = f"{run.name}_{ck.name}"
        proc = subprocess.run([sys.executable, "scripts/eval_hf.py", "--base", args.teacher, "--models", f"{ck.name}={ck}",
                               "--manifests", *manifests, "--name", f"pick_{name}"], cwd=ROOT, capture_output=True, text=True)
        cells = {}
        for line in proc.stdout.splitlines():
            m = CELL.match(line)
            if m:
                _, test, wer, cer, loops, inv, _ = m.groups()
                cells[test] = {"wer": float(wer), "cer": float(cer), "loops": int(loops), "invented": int(inv or 0)}
        if not cells:
            print(proc.stdout[-2000:], proc.stderr[-3000:])
            raise SystemExit(f"eval failed for {ck}")
        scores[ck.name] = cells
        scores_path.write_text(json.dumps(scores, indent=1))
        print(ck.name, json.dumps(cells), flush=True)

    def summary(c):
        invented = sum(c[f"stress/{s}"]["invented"] for s in ("noise", "music", "silence"))
        loops = sum(v["loops"] for v in c.values())
        speech = [v["wer"] for k, v in c.items() if v["wer"] == v["wer"] and not k.startswith("stress/english")]
        return invented, loops, sum(speech) / len(speech)

    rows = []
    for ck, c in scores.items():
        inv, loops, wer = summary(c)
        ok = inv <= args.teacher_invented and loops == 0
        rows.append((not ok, wer, ck, inv, loops, c))
    rows.sort()
    lines = [f"# {run.name}: checkpoints ranked by the stress test", "",
             f"Pass = no loops and invented text on noise+music+silence <= teacher ({args.teacher_invented}). "
             "Then lowest mean WER over the speech stress sets and dev.", "",
             "| ckpt | pass | invented | loops | mean WER | " + " | ".join(STRESS[3:]) + " | dev |", "|---" * (6 + len(STRESS[3:])) + "|"]
    for bad, wer, ck, inv, loops, c in rows:
        per = [f'{c[f"stress/{s}"]["wer"]:.1f}' for s in STRESS[3:]]
        dev = next((v["wer"] for k, v in c.items() if k.endswith("/dev")), float("nan"))
        lines.append(f"| {ck} | {'yes' if not bad else 'no'} | {inv} | {loops} | {wer:.1f} | " + " | ".join(per) + f" | {dev:.1f} |")
    (ROOT / "results" / f"{run.name}_pick.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

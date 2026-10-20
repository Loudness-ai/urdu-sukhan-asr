"""Pick side-by-side examples from cmp_examples.jsonl and write them as Markdown and JSON.

Selection (stated on the pages): clips chosen to show where Urdu Sukhan ASR helps most, taken from the top of a ranking
of all clips by words fixed (base-model errors minus ours), so that each kind of audio is shown once or twice; plus
one of the clips where our model did worse, for balance. The statistics cover every clip, not only the picks.
Wrong words (substituted or inserted, against the reference, after normalisation) are shown in bold.
"""
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else ".")
from textnorm import normalize_for_eval  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
by = defaultdict(dict)
order = []
for r in rows:
    key = r["audio"]
    if key not in by:
        order.append(key)
    by[key][r["model"]] = r
pairs = [by[k] for k in order if {"base-q8", "ours-q8"} <= by[k].keys()]

LANG = lambda r: "en" if r["cat"] == "english" else "ur"  # noqa: E731


def toks(text, lang):
    return [w for w in normalize_for_eval(text, lang).split()]


def wer(ref, hyp, lang):
    r, h = toks(ref, lang), toks(hyp, lang)
    if not r:
        return float(len(h) > 0)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / len(r)


def marked(ref, hyp, lang):
    """The hypothesis with wrong words in bold; a missing word is shown as [–], several as [– N words]. A shown word can normalise to several
    tokens (an apostrophe splits "Committee's"), so the diff runs on tokens and maps back to shown words."""
    shown = hyp.split()
    flat, owner = [], []
    for i, w in enumerate(shown):
        for t in normalize_for_eval(w, lang).split():
            flat.append(t)
            owner.append(i)
    bad, missing_after = set(), {}
    sm = difflib.SequenceMatcher(None, toks(ref, lang), flat, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "insert"):
            bad.update(owner[j] for j in range(j1, j2))
        n = (i2 - i1) if op == "delete" else (i2 - i1) - (j2 - j1) if op == "replace" else 0
        if n > 0:
            at = owner[j2 - 1] if j2 > 0 else -1
            missing_after[at] = missing_after.get(at, 0) + n
    gap = lambda n: "[–]" if n == 1 else f"[– {n} words]"  # noqa: E731
    out = [gap(missing_after[-1])] if -1 in missing_after else []
    for i, w in enumerate(shown):
        out.append(f"**{w}**" if i in bad else w)
        if i in missing_after:
            out.append(gap(missing_after[i]))
    return " ".join(out) if out else "*(nothing)*"


speech = [p for p in pairs if p["base-q8"]["ref"]]
stats = {"clips": len(speech), "better": 0, "worse": 0, "same": 0, "words": 0, "err_base": 0.0, "err_ours": 0.0}
for p in speech:
    lang = LANG(p["base-q8"])
    wb, wo = wer(p["base-q8"]["ref"], p["base-q8"]["hyp"], lang), wer(p["ours-q8"]["ref"], p["ours-q8"]["hyp"], lang)
    n = len(toks(p["base-q8"]["ref"], lang))
    p["wer_base"], p["wer_ours"], p["words"] = wb, wo, n
    stats["words"] += n
    stats["err_base"] += wb * n
    stats["err_ours"] += wo * n
    stats["better" if wo < wb else "worse" if wo > wb else "same"] += 1
stats["wer_base"] = round(100 * stats.pop("err_base") / stats["words"], 1)
stats["wer_ours"] = round(100 * stats.pop("err_ours") / stats["words"], 1)

# (category, file name), in the order shown; the last is one of the clips where our model did worse
PICKS = [("mixed", "2.flac"), ("mixed", "4.flac"), ("short", "9.flac"), ("short", "7.flac"), ("short", "0.flac"),
         ("indicvoices", "tests_indicvoices_1.wav"), ("quiet", "4.flac"), ("phone", "5.flac"),
         ("urduspeech", "tests_urduspeech_2.wav")]
find = {(p["base-q8"]["cat"], Path(p["base-q8"]["audio"]).name): p for p in speech}
chosen = []
for key in PICKS:
    p = find[key]
    chosen.append(("ours worse" if p["wer_ours"] > p["wer_base"] else "ours better" if p["wer_ours"] < p["wer_base"]
                   else "same", p))
nospeech = [p for p in pairs if not p["base-q8"]["ref"]]

out = {"stats": stats, "examples": [], "nospeech": []}
for why, p in chosen:
    b, o = p["base-q8"], p["ours-q8"]
    lang = LANG(b)
    out["examples"].append({"cat": b["cat"], "why": why, "audio": b["audio"], "ref": b["ref"],
                            "base": b["hyp"], "ours": o["hyp"], "base_marked": marked(b["ref"], b["hyp"], lang),
                            "ours_marked": marked(o["ref"], o["hyp"], lang),
                            "wer_base": round(100 * p["wer_base"], 1), "wer_ours": round(100 * p["wer_ours"], 1),
                            "words": p["words"], "err_base": round(p["wer_base"] * p["words"]),
                            "err_ours": round(p["wer_ours"] * p["words"])})
for p in nospeech:
    b, o = p["base-q8"], p["ours-q8"]
    out["nospeech"].append({"cat": b["cat"], "audio": b["audio"], "base": b["hyp"], "ours": o["hyp"],
                            "ours_nsb": o.get("hyp_nsb", "")})
json.dump(out, open("examples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(stats))
for e in out["examples"]:
    print(f"\n[{e['cat']} / {e['why']}] base {e['wer_base']} ours {e['wer_ours']}\n REF  {e['ref']}\n BASE {e['base_marked']}\n OURS {e['ours_marked']}")
for e in out["nospeech"]:
    print(f"\n[{e['cat']}] BASE [{e['base']}] OURS [{e['ours']}] OURS-NSB [{e['ours_nsb']}]")

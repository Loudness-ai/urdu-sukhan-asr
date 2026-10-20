"""Corrections learned from proofreading, applied to drafts that haven't been proofread yet.

Each project can have projects/<name>/corrections.tsv:

    wrong<TAB>right<TAB>optional note
    [only: training 1]
    مصبت	مثبت	the book's theme word; the model never gets it right

Only list fixes that are right in every context (misspellings, names). Words that
are real but sometimes wrong (تھی/تھیں, ہاتھ/ہار) stay with the proofreader.
Two built-in rules also run for Urdu: Western digits -> Urdu digits, and spoken
years (انیس سو چھیاسی) -> digits (۱۹۸۶), matching how our proofreader writes them.
"""

import difflib
import re
from collections import Counter
from pathlib import Path

from .text import clean_label

URDU_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

_NUMBERS = """ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس بیس
اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس تیس اکتیس بتیس تینتیس چونتیس پینتیس چھتیس
سینتیس اڑتیس انتالیس چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس سینتالیس اڑتالیس انچاس
پچاس اکاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ ساٹھ اکسٹھ باسٹھ ترسٹھ چونسٹھ پینسٹھ چھیاسٹھ
سڑسٹھ اڑسٹھ انہتر ستر اکہتر بہتر تہتر چوہتر پچھتر چھہتر ستتر اٹھہتر اناسی اسی اکیاسی بیاسی تراسی
چوراسی پچاسی چھیاسی ستاسی اٹھاسی نواسی نوے اکانوے بانوے ترانوے چورانوے پچانوے چھیانوے ستانوے
اٹھانوے ننانوے""".split()
# Spellings the model produces for some of them.
_NUMBER_VARIANTS = {"اکاسی": 81, "بیانوے": 92, "نینانوے": 99, "چھے": 6, "اٹھتر": 78, "تریتر": 73, "چورتر": 74}


def _num_key(w):
    return w.replace("ھ", "")  # the model often drops ھ (چیاسی for چھیاسی)


NUMBER_WORDS = {_num_key(w): i + 1 for i, w in enumerate(_NUMBERS)}
NUMBER_WORDS.update({_num_key(w): n for w, n in _NUMBER_VARIANTS.items()})
CENTURIES = {"انیس": 19, "اٹھارہ": 18}


def load_rules(path):
    """Rules as (wrong words, right, scope). A line '[only: <text>]' limits the rules below it to
    recordings whose file name contains <text> (one book's names, one narrator's accent);
    '[all]' switches back."""
    rules, scope = [], None
    if Path(path).exists():
        for line in Path(path).read_text("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                scope = line[1:-1].removeprefix("only:").strip() or None
                scope = None if scope == "all" else scope
                continue
            wrong, right = line.split("\t")[:2]
            rules.append((wrong.split(), right.strip(), scope))
    return sorted(rules, key=lambda r: -len(r[0]))  # longest phrase wins


def _year_at(tokens, i):
    """'انیس سو چھیاسی' at tokens[i] -> ('۱۹۸۶', 3)."""
    if i + 2 < len(tokens) and tokens[i] in CENTURIES and tokens[i + 1] == "سو":
        n = NUMBER_WORDS.get(_num_key(tokens[i + 2]))
        if n:
            return f"{CENTURIES[tokens[i]]}{n:02d}".translate(URDU_DIGITS), 3
    return None


def apply(words, rules, lang, source=""):
    """Apply learned fixes to a draft's word spans. Fixed spans get p=1 and 'fixed' = what the model wrote.

    A span can hold several words (house style turns کیلئے into 'کے لیے'), so matching runs on
    individual words but only replaces whole spans.
    """
    if lang != "ur" or not words:
        return words
    toks, owner = [], []
    for si, w in enumerate(words):
        for t in w["w"].split():
            toks.append(t)
            owner.append(si)
    out, i = [], 0
    done_spans = set()
    while i < len(toks):
        hit = _year_at(toks, i)
        if not hit:
            for wrong, right, scope in rules:
                if scope and scope not in source:
                    continue
                if toks[i : i + len(wrong)] == wrong:
                    hit = (right, len(wrong))
                    break
        if not hit and re.search(r"[0-9]", toks[i]):
            hit = (toks[i].translate(URDU_DIGITS), 1)
        s0 = owner[i]
        if hit:
            n = hit[1]
            s1 = owner[i + n - 1]
            whole = (i == 0 or owner[i - 1] != s0) and (i + n == len(toks) or owner[i + n] != s1)
            if whole and s0 not in done_spans:
                was = " ".join(words[s]["w"] for s in range(s0, s1 + 1))
                out.append({"w": hit[0], "p": 1.0, "start": words[s0]["start"], "end": words[s1]["end"], "fixed": was})
                done_spans.update(range(s0, s1 + 1))
                i += n
                continue
        if s0 not in done_spans:
            out.append(words[s0])
            done_spans.add(s0)
        i += 1
    return out


def suggest(items, rules, lang="ur"):
    """Substitutions the proofreader made, with how safe they look. For growing corrections.tsv."""
    known = {" ".join(r[0]) for r in rules}
    subs = Counter()
    done = [it for it in items if it["status"] == "done" and it.get("text")]
    for it in done:
        d, t = clean_label(it["draft"], lang).split(), clean_label(it["text"], lang).split()
        for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, d, t, autojunk=False).get_opcodes():
            if op == "replace" and a2 - a1 <= 3 and b2 - b1 <= 3:
                subs[(" ".join(d[a1:a2]), " ".join(t[b1:b2]))] += 1
    yours = " " + " ".join(clean_label(it["text"], lang) for it in done) + " "
    todo = [" " + clean_label(it["draft"], lang) + " " for it in items if it["status"] == "todo" and not it.get("text")]
    rows = []
    for (a, b), n in subs.items():
        if a in known or _year_at(a.split(), 0):  # spoken years are handled by the built-in rule
            continue
        # 'kept' > 0 means the proofreader also left this word as-is somewhere: it depends on context.
        rows.append({"wrong": a, "right": b, "fixed": n, "kept": yours.count(f" {a} "),
                     "todo_clips": sum(f" {a} " in t for t in todo)})
    return sorted(rows, key=lambda r: (r["kept"] > 0, -r["todo_clips"], -r["fixed"]))

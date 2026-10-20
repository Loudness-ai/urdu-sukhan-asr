"""Text normalization for training labels and for scoring.

Two levels:
  clean_label(text, lang)  -> cleanup applied to training targets and to drafts:
                              fixes encoding noise, applies the Urdu house style
                              and (for now) drops punctuation.
  normalize_for_eval(text, lang) -> aggressive normalization used before WER/CER,
                              so spelling variants that mean the same thing
                              do not count as errors.
"""

import re
import unicodedata

# Arabic-script languages that Whisper supports and that share these letter variants.
ARABIC_SCRIPT = {"ur", "fa", "ps", "sd", "ar", "ug"}
# Languages where Arabic letter forms (ي ك ه) are the correct ones; we must not
# rewrite them into Urdu/Persian forms.
ARABIC_FORMS_ARE_CORRECT = {"ar", "ug"}

# Arabic code points that Urdu text often contains by mistake (copy-paste from
# Arabic keyboards / OCR). Mapped to the Urdu letters they stand for.
URDU_LETTER_MAP = {
    "ي": "ی",  # ي Arabic yeh        -> ی Farsi yeh
    "ى": "ی",  # ى alef maksura      -> ی
    "ك": "ک",  # ك Arabic kaf        -> ک keheh
    "ه": "ہ",  # ه Arabic heh        -> ہ heh goal
    "ە": "ہ",  # ە ae                -> ہ
    "ة": "ۃ",  # ة teh marbuta       -> ۃ
}
# Persian/Pashto/Sindhi use Farsi yeh and keheh but keep ه.
PERSIAN_LETTER_MAP = {
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
}

TATWEEL = "ـ"
# Invisible characters that carry no meaning in a transcript (zero-width space, BOM, direction marks).
INVISIBLE = dict.fromkeys(map(ord, "​‎‏‪‫‬‭‮⁦⁧⁨⁩﻿"), None)
# ZWNJ/ZWJ do matter in Urdu/Persian spelling, so only scoring drops them.
ZERO_WIDTH = dict.fromkeys(map(ord, "‌‍"), None)

# Harakat / tashkeel and other combining marks used on Arabic script.
ARABIC_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")

# Eval-only: letters carrying a hamza that transcribers write either way.
EVAL_HAMZA_FOLD = {
    "ۂ": "ہ",  # ۂ -> ہ
    "ۓ": "ے",  # ۓ -> ے
}

DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩"
    "۰۱۲۳۴۵۶۷۸۹"
    "०१२३४५६७८९",
    "0123456789" * 3,
)

_SPACES = re.compile(r"\s+")

# --- Urdu house style -------------------------------------------------------
# Picked by our proofreader (Oct 2026): ی-forms of لینا/دینا/کرنا, ہ in pronoun
# forms, verb + helper verb written as two words, and a few fixed spellings.
# Applied to every training label, to drafts shown for proofreading, and to both
# sides before scoring, so the model learns one spelling for each word.
URDU_STYLE_WORDS = {
    "لئے": "لیے", "لیئے": "لیے",
    "کیلئے": "کے لیے", "کیلیے": "کے لیے", "کےلیے": "کے لیے", "کےلئے": "کے لیے",
    "اسلئے": "اس لیے", "اسلیے": "اس لیے",
    "دئے": "دیے", "دیئے": "دیے", "دئیے": "دیے",
    "کئے": "کیے", "کیئے": "کیے", "کئیے": "کیے", "لئیے": "لیے",
    "چاہئے": "چاہیے", "چاہیئے": "چاہیے",
    "انھیں": "انہیں", "انھوں": "انہوں", "انھی": "انہی",
    "جنھیں": "جنہیں", "جنھوں": "جنہوں",
    "تمھیں": "تمہیں", "تمھارا": "تمہارا", "تمھاری": "تمہاری", "تمھارے": "تمہارے",
    "یوروپ": "یورپ",
    "روئیہ": "رویہ", "روئیوں": "رویوں",
    "پانچوی": "پانچویں", "ساتوی": "ساتویں", "آٹھوی": "آٹھویں", "آٹوی": "آٹھویں", "نووی": "نویں", "دسوی": "دسویں", "گیارہوی": "گیارہویں", "گیاروی": "گیارہویں", "بارہوی": "بارہویں", "باروی": "بارہویں", "تیرہوی": "تیرہویں", "تیروی": "تیرہویں", "چودہوی": "چودہویں", "چودھوی": "چودھویں", "پندرہوی": "پندرہویں", "سولہوی": "سولہویں", "سترہوی": "سترہویں", "اٹھارہوی": "اٹھارہویں", "انیسوی": "انیسویں", "بیسوی": "بیسویں",
    "چھے": "چھ", "بالاخر": "بالآخر", "اعلٰی": "اعلیٰ", "اسپتال": "ہسپتال", "دباو": "دباؤ",
    "افزاء": "افزا", "فورا": "فوراً", "فوراں": "فوراً", "روئیے": "رویے", "درامد": "درآمد", "برامد": "برآمد",
    "کاروائی": "کارروائی", "ہوے": "ہوئے", "کاروائیاں": "کارروائیاں", "کاروائیوں": "کارروائیوں",
}
URDU_STYLE_PHRASES = {"در اصل": "دراصل", "لا شعوری": "لاشعوری", "قطعہ نظر": "قطع نظر", "نکتہ نظر": "نقطہ نظر", "برائے کرم": "براہ کرم", "لا جواب": "لاجواب", "السلام و علیکم": "السلام علیکم", "السلام وعلیکم": "السلام علیکم", "لا پرواہ": "لاپرواہ", "لا پرواہی": "لاپرواہی", "لا پروائی": "لاپروائی"}
# Verb stem + helper written as one word (کرکے, ہوگئی, جاسکتا) -> two words.
# Future endings (گا/گی/گے) are left alone: ہوگا/ہو گا has no agreed form, and جاگے is a word.
_STEMS = ("کر", "ہو", "جا", "دے", "لے", "آ", "رہ", "دیکھ", "کہہ", "پا")
_HELPERS = ("سکتا", "سکتی", "سکتے", "سکا", "سکی", "سکے", "گیا", "گئی", "گئے", "چکا", "چکی", "چکے",
            "دیا", "دی", "دیے", "لیا", "لی", "لیے", "رہا", "رہی", "رہے")
_REAL_WORDS = {"ہولی", "جالی", "آلی", "پالی", "پالیا"}  # look like stem+helper but are words of their own
for _s in _STEMS:
    for _h in _HELPERS:
        if _s + _h not in _REAL_WORDS:
            URDU_STYLE_WORDS.setdefault(_s + _h, f"{_s} {_h}")
URDU_STYLE_WORDS["کرکے"] = "کر کے"

# A word includes its Arabic diacritics, so a rule like فورا -> فوراً can't fire again on فوراً.
_WORD = re.compile(r"(?:[^\W\d_]|[\u0610-\u061A\u064B-\u065F\u0670])+")
_REPEATED_MARK = re.compile(r"([\u0610-\u061A\u064B-\u065F\u0670])\1+")


# Training text carries no punctuation for now (decided Oct 2026): our proofread clips
# have almost none and public sets have plenty, and a mix teaches the model to punctuate
# at random. Kept: apostrophes/hyphens inside words (don't, e-mail), decimal and
# thousands separators between digits (3.5, 1,000), and percent signs.
KEEP_PUNCTUATION = False
_PUNCT_KEEP = re.compile(r"(?<=\w)['’\-](?=\w)|(?<=\d)[.,٫٬](?=\d)|[%٪]")


def strip_punctuation(text):
    kept = {m.start() for m in _PUNCT_KEEP.finditer(text)}
    return "".join(" " if unicodedata.category(c)[0] == "P" and i not in kept else c for i, c in enumerate(text))


URDU_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def urdu_house_style(text):
    text = text.translate(URDU_DIGITS)  # the proofreader writes ۱۹۸۱, not 1981
    text = _WORD.sub(lambda m: URDU_STYLE_WORDS.get(m.group(0), m.group(0)), text)
    for a, b in URDU_STYLE_PHRASES.items():
        text = re.sub(rf"(?<!\w){a}(?!\w)", b, text)
    return text


def _letter_map(lang):
    if lang == "ur":
        return URDU_LETTER_MAP
    if lang in ARABIC_SCRIPT and lang not in ARABIC_FORMS_ARE_CORRECT:
        return PERSIAN_LETTER_MAP
    return None


def clean_label(text, lang):
    """Light cleanup for training targets: fix encoding, keep the words as written."""
    text = unicodedata.normalize("NFC", text).translate(INVISIBLE)
    text = _REPEATED_MARK.sub(r"\1", text)  # e.g. فوراًًً -> فوراً
    mapping = _letter_map(lang)
    if mapping:
        text = text.replace(TATWEEL, "")
        text = "".join(mapping.get(c, c) for c in text)
    if lang == "ur":
        text = urdu_house_style(text)
    if not KEEP_PUNCTUATION:
        text = strip_punctuation(text)
    return _SPACES.sub(" ", text).strip()


def normalize_for_eval(text, lang):
    """Aggressive normalization before scoring: no punctuation, case, diacritics or digit-system differences."""
    text = clean_label(text, lang)
    text = text.translate(ZERO_WIDTH).translate(DIGITS).lower()
    if lang in ARABIC_SCRIPT:
        text = "".join(EVAL_HAMZA_FOLD.get(c, c) for c in text)
        text = ARABIC_DIACRITICS.sub("", text)
    # Punctuation and symbols become spaces (covers ۔ ، ؟ ؛ « » as well as ASCII).
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return _SPACES.sub(" ", text).strip()


def word_char_errors(refs, hyps, lang):
    """Corpus WER and CER (percent) after normalization."""
    import jiwer

    refs_n = [normalize_for_eval(r, lang) for r in refs]
    hyps_n = [normalize_for_eval(h, lang) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs_n, hyps_n) if r]
    refs_n, hyps_n = [p[0] for p in pairs], [p[1] for p in pairs]
    wer = jiwer.wer(refs_n, hyps_n) * 100
    # CER on text without spaces, so word-boundary disagreements (common in Urdu) count once, not twice.
    cer = jiwer.cer([r.replace(" ", "") for r in refs_n], [h.replace(" ", "") for h in hyps_n]) * 100
    return wer, cer

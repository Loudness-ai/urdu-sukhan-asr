"""Write the "Examples" section of the Hugging Face card and the GitHub README from examples.json.

    python3 make_pages_section.py HF_README GITHUB_README

The section replaces whatever stands between <!-- EXAMPLES START --> and <!-- EXAMPLES END --> in each file.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
s = d["stats"]
AUDIO = "https://huggingface.co/datasets/Loudness-ai/urdu-sukhan-asr-examples/blob/main/examples/{}.mp3"
NAMES = ["01-urdu-english-1", "02-urdu-english-2", "03-reply-alag-alag", "04-reply-hello", "05-reply-eighty-six",
         "06-indicvoices-ji-haan", "07-quiet-speech", "08-phone-line", "09-urduspeech-train-name"]
KIND = {"short": "One- or two-word reply (IndicVoices)", "cv": "Common Voice, speaker not seen in training",
        "indicvoices": "IndicVoices, conversational", "fleurs": "FLEURS, read news", "urduspeech": "UrduSpeech",
        "quiet": "Quiet speech (a Common Voice clip, 30 dB softer)", "phone": "Phone line (a Common Voice clip)",
        "mixed": "Urdu then English (Common Voice + FLEURS)"}
MEANING = [
    "“It is right in this”, then an English sentence. The base model wrote four English words for the whole clip; ours "
    "wrote both halves and got the last two words wrong.",
    "“What is the accountability court's verdict?”, then an English sentence. The base model translated the Urdu into "
    "English, wrongly, instead of writing it down; ours wrote both halves without a mistake.",
    "“separately”. The base model wrote “Allah, Allah”.",
    "“hello”. The base model wrote “music”.",
    "“eighty-six”",
    "“yes, yes, yes, yes”",
    "“Tom survived the car accident.” Both models wrote the name Tom as ٹوم.",
    "“the freedom that one sees in this atmosphere”",
    "“Gaya – Anand Vihar Garib Rath Express” (a train). Ours wrote “Anand Vihar” as one word.",
]
assert len(d["examples"]) == len(NAMES) == len(MEANING)

RED = "#d32f2f"


def red(marked):
    """Wrong words (**word**) and missing words ([–], [– N words]) in red; GitHub drops the colour and keeps bold."""
    out = re.sub(r"\*\*(.+?)\*\*", lambda m: f'<b style="color:{RED}">{m.group(1)}</b>', marked)
    return re.sub(r"\[– ?[^\]]*\]", lambda m: f'<b style="color:{RED}">{m.group(0)}</b>', out)


def result(mine, other):
    """✅ for the model with fewer mistakes on the clip, ❌ for the one with more, then its count."""
    mark = "✅" if mine < other else "❌" if mine > other else "➖"
    return f"{mark} " + ("no mistakes" if mine == 0 else f"{mine} mistake" + ("" if mine == 1 else "s"))


def intro(colour):
    wrong = "Red" if colour else "**Bold**"
    return (
        "Public test clips transcribed by Whisper large-v3-turbo and by Urdu Sukhan ASR v1, run the way an application "
        f"runs them (whisper.cpp, q8_0, beam 5, temperature fallback, no timestamps). {wrong} marks a wrong word, [–] a "
        "missing one and [– N words] several, against the test set's reference after the same normalisation as our "
        "scores, so spelling and spacing count. ✅ marks the model with fewer mistakes on the clip and ❌ the one with "
        "more; a mistake is a wrong, extra or missing word. We picked these clips by hand to show where our model helps "
        "most, mostly from the top of a ranking of all clips by the number of words it fixed; the last example is one of "
        f"the four clips where it did worse. Over all {s['clips']} clips with speech that we ran this way, our model was "
        f"better on {s['better']}, worse on {s['worse']} and the same on {s['same']} ({s['wer_ours']}% against "
        f"{s['wer_base']}% words wrong).")


ns = {r["cat"]: r for r in d["nospeech"]}
rows = [json.loads(l) for l in (HERE / "cmp_examples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
base_nsb = {r["cat"]: r.get("hyp_nsb", "") for r in rows if r["model"] == "base-q8" and r["cat"] in ("silence", "noise")}


def said(text, gloss=""):
    """A no-speech transcript: any word is a mistake (red, ❌); nothing at all is right (✅)."""
    if not text:
        return "✅ *(nothing)*"
    return f'❌ <b style="color:{RED}">{text}</b>' + (f" (“{gloss}”)" if gloss else "")


CREDITS = ("Clips: Common Voice (CC0), FLEURS (CC BY 4.0, Google), IndicVoices and UrduSpeech/Rasa (CC BY 4.0, "
           "AI4Bharat), MUSAN (CC BY 4.0); the quiet, phone and Urdu-then-English clips are Common Voice and FLEURS clips "
           "that we turned down, passed through a phone filter or joined, and the hiss is generated.")
NOSPEECH = ("On audio without speech, every Whisper decoder by default must write at least one word. Allowing an empty "
            "result (`suppress_blank = false`, or `-nsb` with our patch) only helps a model that has learned to stop")

# GitHub shows no <audio> players and no coloured text, but it plays videos uploaded to it: each clip is also a short
# video (a title card with the reference text, and the audio), listed in github_videos.json.
VIDEO = json.loads((HERE / "github_videos.json").read_text(encoding="utf-8"))
RAW = "https://huggingface.co/datasets/Loudness-ai/urdu-sukhan-asr-examples/resolve/main/examples/{}.mp3"


def blocks(colour, player):
    """The Examples section: one block per clip with its player, transcripts and result."""
    out = ["## Examples", "", intro(colour)]
    for k, (name, e, meaning) in enumerate(zip(NAMES, d["examples"], MEANING), 1):
        kind = KIND[e["cat"]] + (" · one of the four clips where ours did worse" if e["why"] == "ours worse" else "")
        verdict = ("✅ Urdu Sukhan ASR better" if e["err_ours"] < e["err_base"] else
                   "❌ Urdu Sukhan ASR worse" if e["err_ours"] > e["err_base"] else "➖ same")
        out += ["", f"**{k}. {kind}** · {verdict}", "", player(name), "",
                "| | Transcript | Result |", "|---|---|---|",
                f"| Reference | {e['ref']} | {e['words']} words |", f"| Meaning | {meaning} | |",
                f"| Whisper large-v3-turbo | {red(e['base_marked'])} | {result(e['err_base'], e['err_ours'])} |",
                f"| **Urdu Sukhan ASR v1** | {red(e['ours_marked'])} | {result(e['err_ours'], e['err_base'])} |"]
    out += ["", f"**No speech**: {NOSPEECH.lower()[0] + NOSPEECH[1:]}. Any word here is a mistake.", ""]
    for name, cat, label in (("10-faint-hiss", "silence", "Faint hiss and hum"), ("11-noise", "noise", "Noise (MUSAN)")):
        out += [f"*{label}*", "", player(name), "",
                "| | default | blank allowed |", "|---|---|---|",
                f"| Whisper large-v3-turbo | {said(ns[cat]['base'], 'music')} | {said(base_nsb[cat])} |",
                f"| **Urdu Sukhan ASR v1** | {said(ns[cat]['ours'], 'all right')} | {said(ns[cat]['ours_nsb'])} |", ""]
    out.append(CREDITS)
    return "\n".join(out)


players = blocks(True, lambda n: f'<audio controls preload="none" src="{RAW.format(n)}"></audio>')
section = blocks(False, lambda n: VIDEO[n])

for path in sys.argv[1:]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    body = players if "huggingface" in str(p.parent) or p.parent.name == "hf" else section
    new = re.sub(r"<!-- EXAMPLES START -->.*?<!-- EXAMPLES END -->",
                 lambda m: "<!-- EXAMPLES START -->\n" + body + "\n<!-- EXAMPLES END -->", text, flags=re.S)
    p.write_text(new, encoding="utf-8")
    print("updated", path, "(players)" if body is players else "(table)")

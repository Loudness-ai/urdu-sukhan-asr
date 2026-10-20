"""Transcribe public test clips with one whisper.cpp model, for side-by-side examples (Urdu Sukhan ASR v1 vs the base model).

    python3 compare_examples.py THREADS OUT.jsonl

Candidates come only from public test sets (Common Voice, FLEURS, IndicVoices, UrduSpeech, MUSAN) and the stress sets
built from them; clips built from the author's audiobooks are left out. Settings as an application runs the model:
beam 5, temperature fallback, no timestamps, no carried context; language set to Urdu, or auto for English and
Urdu-then-English clips. No-speech clips are also run with -nsb (blank allowed).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/opt/urdu-stt-test")
B = Path("/opt/backups/whisper-lite-2026-10-20/data")
CLI = ROOT / "whisper.cpp-nsb/build/bin/whisper-cli"
SAMPLES = ROOT / "tools/testpage/samples"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


book2 = {r["text"] for r in jl(B / "proofread/novel/test_book2.jsonl")}
cands = []
for s in json.load(open(ROOT / "tools/testpage/samples.json", encoding="utf-8")):
    cat = {"Common Voice, new speakers": "cv", "IndicVoices": "indicvoices", "FLEURS Urdu": "fleurs",
           "UrduSpeech": "urduspeech", "English (FLEURS)": "english"}.get(s["set"])
    if cat and sum(1 for c in cands if c["cat"] == cat) < {"cv": 5, "indicvoices": 5, "fleurs": 4, "urduspeech": 3, "english": 2}[cat]:
        cands.append({"cat": cat, "audio": str(SAMPLES / s["file"]), "ref": s["text"],
                      "lang": "auto" if cat == "english" else "ur"})
for name, n, lang in (("short", 10, "ur"), ("quiet", 6, "ur"), ("phone", 6, "ur"), ("mixed", 6, "auto")):
    k = 0
    for r in jl(B / f"stress/{name}.jsonl"):
        urdu = r["text"].split(" ")
        if name in ("quiet", "phone") and r["text"] in book2:
            continue
        if name == "mixed" and any(r["text"].startswith(t) for t in book2):
            continue
        if not r["text"].strip() or r["text"].startswith("<"):
            continue
        cands.append({"cat": name, "audio": str(B / "stress" / r["audio"]), "ref": r["text"], "lang": lang})
        k += 1
        if k == n:
            break
for name, ids in (("silence", [12]), ("noise", [0])):
    rows = jl(B / f"stress/{name}.jsonl")
    for i in ids:
        cands.append({"cat": name, "audio": str(B / "stress" / rows[i]["audio"]), "ref": "", "lang": "ur"})


def run(model_file, audio, lang, threads, extra=()):
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"  # private copy: whisper.cpp writes next to its input
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", audio, "-ar", "16000", "-ac", "1", str(wav)], check=True)
        p = subprocess.run(["nice", "-n", "10", str(CLI), "-m", model_file, "-l", lang, "-bs", "5", "-nt", "-mc", "0",
                            "-t", str(threads), *extra, "-f", str(wav)], capture_output=True, text=True)
        return " ".join(p.stdout.split())


if __name__ == "__main__":
    # both models on each clip in turn, so partial results already pair up
    models = [("base-q8", str(ROOT / "models/ggml/ggml-large-v3-turbo-q8_0.bin")),
              ("ours-q8", str(ROOT / "models/ggml/ggml-r6c2775-q8_0.bin"))]
    threads, out = int(sys.argv[1]), sys.argv[2]
    with open(out, "w", encoding="utf-8") as f:
        for c in cands:
            for model_id, model_file in models:
                row = dict(c, model=model_id, hyp=run(model_file, c["audio"], c["lang"], threads))
                if c["cat"] in ("silence", "noise", "music"):
                    row["hyp_nsb"] = run(model_file, c["audio"], c["lang"], threads, ("-nsb",))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    print("COMPARE_DONE", len(cands))

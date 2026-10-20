# mac_cv

Manifest: `data/cv20k/test_unique.jsonl` — 260 clips, 18.3 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| r6c2775-q8_0 | 874 MB | 27.9 | 16.5 | 1.7 |
| blend-r4r5-q8_0 | 874 MB | 27.3 | 16.0 | 1.0 |
| large-v3-turbo-q8_0 | 874 MB | 38.1 | 23.7 | 1.3 |
| novel-r5-q8_0 | 874 MB | 27.1 | 15.4 | 0.8 |
| novel-r4-q8_0 | 874 MB | 30.6 | 17.6 | 1.4 |

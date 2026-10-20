# mac_book1_r5

Manifest: `data/proofread/novel/test_book1.jsonl` — 48 clips, 13.4 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| r6c2775-q8_0 | 874 MB | 4.2 | 1.7 | 5.2 |
| blend-r4r5-q8_0 | 874 MB | 3.6 | 1.4 | 4.0 |
| novel-r5-q8_0 | 874 MB | 3.5 | 1.4 | 4.1 |

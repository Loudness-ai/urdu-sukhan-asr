# r4_book1

Manifest: `data/proofread/novel/test_book1.jsonl` — 48 clips, 13.4 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r4-q4_0 | 474 MB | 4.5 | 1.7 | 3.6 |
| novel-r4-q8_0 | 874 MB | 3.8 | 1.5 | 3.1 |
| novel-r3-q8_0 | 874 MB | 4.3 | 1.9 | 3.5 |

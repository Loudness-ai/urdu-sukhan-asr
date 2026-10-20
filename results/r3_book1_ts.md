# r3_book1_ts

Manifest: `data/proofread/novel/test_book1.jsonl` — 48 clips, 13.4 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r2-q8_0 | 874 MB | 7.9 | 5.9 | 6.2 |
| novel-r3-q8_0 | 874 MB | 4.3 | 1.8 | 6.7 |

# r4_book2

Manifest: `data/proofread/novel/test_book2.jsonl` — 10 clips, 2.9 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r4-q4_0 | 474 MB | 13.6 | 4.8 | 2.8 |
| novel-r4-q8_0 | 874 MB | 11.7 | 3.9 | 2.7 |
| novel-r3-q8_0 | 874 MB | 13.2 | 4.7 | 3.3 |

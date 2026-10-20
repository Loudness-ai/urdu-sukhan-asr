# r3_book1

Manifest: `data/proofread/novel/test_book1.jsonl` — 48 clips, 13.4 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r3-q4_0 | 474 MB | 5.0 | 2.0 | 6.4 |
| novel-r3-q5_0 | 574 MB | 4.8 | 1.8 | 5.6 |
| large-v3-turbo-q8_0 | 874 MB | 5.9 | 2.6 | 6.1 |
| novel-r1-q8_0 | 874 MB | 3.9 | 1.4 | 6.1 |
| novel-r2-q8_0 | 874 MB | 3.9 | 1.5 | 6.2 |
| novel-r3-q8_0 | 874 MB | 4.3 | 1.9 | 5.8 |

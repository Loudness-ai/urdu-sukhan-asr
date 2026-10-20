# r3_book2

Manifest: `data/proofread/novel/test_book2.jsonl` — 10 clips, 2.9 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r3-q4_0 | 474 MB | 15.4 | 5.4 | 5.9 |
| large-v3-turbo-q8_0 | 874 MB | 17.0 | 6.3 | 4.4 |
| novel-r1-q8_0 | 874 MB | 15.1 | 5.7 | 5.4 |
| novel-r2-q8_0 | 874 MB | 15.7 | 5.8 | 5.4 |
| novel-r3-q8_0 | 874 MB | 13.2 | 4.7 | 5.5 |

# heldout_r2

Manifest: `data/proofread/novel/test.jsonl` — 40 clips, 11.6 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r2-q4_0 | 474 MB | 4.7 | 2.5 | 3.8 |
| novel-r2-q5_0 | 574 MB | 9.6 | 7.3 | 3.5 |
| large-v3-turbo-q8_0 | 874 MB | 5.6 | 2.8 | 3.7 |
| novel-r1-q8_0 | 874 MB | 2.7 | 1.2 | 3.5 |
| novel-r2-q8_0 | 874 MB | 4.8 | 2.9 | 3.6 |

# fleurs_r2

Manifest: `data/fleurs/ur_pk/test.jsonl` — 80 clips, 12.6 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r2-q4_0 | 474 MB | 20.7 | 8.8 | 2.4 |
| large-v3-turbo-q8_0 | 874 MB | 20.9 | 8.5 | 2.2 |
| novel-r1-q8_0 | 874 MB | 20.1 | 8.2 | 2.3 |
| novel-r2-q8_0 | 874 MB | 20.1 | 8.4 | 2.4 |

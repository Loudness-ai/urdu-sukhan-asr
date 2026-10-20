# fleurs_r3

Manifest: `data/fleurs/ur_pk/test.jsonl` — 80 clips, 12.6 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r3-q4_0 | 474 MB | 21.0 | 8.4 | 3.6 |
| novel-r3-q8_0 | 874 MB | 19.9 | 8.0 | 3.6 |

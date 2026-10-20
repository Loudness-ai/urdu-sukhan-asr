# fleurs_ur_baseline

Manifest: `data/fleurs/ur_pk/test.jsonl` — 299 clips, 48.9 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| small-q8_0 | 264 MB | 36.6 | 15.1 | 6.9 |
| large-v3-turbo-q8_0 | 874 MB | 21.0 | 8.2 | 3.3 |

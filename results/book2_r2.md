# book2_r2

Manifest: `data/proofread/novel/book2_done.jsonl` — 47 clips, 14.2 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| large-v3-turbo-q8_0 | 874 MB | 9.1 | 4.0 | 3.4 |
| novel-r1-q8_0 | 874 MB | 8.0 | 3.7 | 3.4 |
| novel-r2-q8_0 | 874 MB | 8.7 | 4.2 | 3.5 |

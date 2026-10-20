# heldout_r2_nt

Manifest: `data/proofread/novel/test.jsonl` — 40 clips, 11.6 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| novel-r2-q4_0 | 474 MB | 3.8 | 1.5 | 4.0 |
| novel-r2-q5_0 | 574 MB | 3.6 | 1.5 | 3.9 |
| large-v3-turbo-q8_0 | 874 MB | 5.5 | 2.5 | 4.2 |
| novel-r1-q8_0 | 874 MB | 2.7 | 1.1 | 4.1 |
| novel-r2-q8_0 | 874 MB | 3.2 | 1.4 | 3.9 |

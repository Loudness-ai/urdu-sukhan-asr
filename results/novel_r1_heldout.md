# novel_r1_heldout

Manifest: `data/proofread/novel/test.jsonl` — 24 clips, 7.0 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| large-v3-turbo-q8_0 | 874 MB | 6.3 | 3.7 | 7.7 |
| novel-r1-q8_0 | 874 MB | 5.6 | 3.1 | 5.4 |

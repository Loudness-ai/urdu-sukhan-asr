# mac_book2_r5

Manifest: `data/proofread/novel/test_book2.jsonl` — 10 clips, 2.9 min, language `ur`.
whisper.cpp, beam size 5, Apple GPU (Metal). Speed includes model loading.
CER ignores spaces, so a word-boundary disagreement counts once. Lower is better for WER/CER.

| Model | Size | WER % | CER % | Speed (x real time) |
|---|---:|---:|---:|---:|
| r6c2775-q8_0 | 874 MB | 12.0 | 4.1 | 3.7 |
| blend-r4r5-q8_0 | 874 MB | 10.3 | 3.5 | 2.9 |
| novel-r5-q8_0 | 874 MB | 10.6 | 3.6 | 3.3 |

# mac_vad_r6

whisper.cpp, beam 5 with temperature fallback, no timestamps (as the app runs). Each cell: word / letter error rate, loops, invented text on no-speech clips.

| Test set | r6c2775-q8_0 |
|---|---:|
| stress/noise (40) | loops 0 · invented 2/40 |
| stress/music (30) | loops 0 · invented 0/30 |
| stress/silence (30) | loops 0 · invented 0/30 |
| stress/quiet (40) | 25.8 / 18.2 · loops 0 |
| stress/phone (40) | 25.0 / 15.2 · loops 0 |
| stress/short (40) | 23.6 / 27.6 · loops 0 |
| stress/mixed (20) | 45.5 / 42.6 · loops 0 |
| novel/test_book1 (48) | 4.3 / 1.8 · loops 0 |
| novel/test_book2 (10) | 12.6 / 4.5 · loops 0 |
| cv20k/test_unique (260) | 28.7 / 18.0 · loops 0 |

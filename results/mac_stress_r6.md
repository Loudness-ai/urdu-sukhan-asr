# mac_stress_r6

whisper.cpp, beam 5 with temperature fallback, no timestamps (as the app runs). Each cell: word / letter error rate, loops, invented text on no-speech clips.

| Test set | blend-r4r5-q8_0 | r6c2775-q8_0 |
|---|---:|---:|
| stress/noise (40) | loops 0 · invented 9/40 | loops 0 · invented 10/40 |
| stress/music (30) | loops 1 · invented 17/30 | loops 0 · invented 0/30 |
| stress/silence (30) | loops 0 · invented 11/30 | loops 0 · invented 6/30 |
| stress/quiet (40) | 24.2 / 15.4 · loops 0 | 24.2 / 15.3 · loops 2 |
| stress/phone (40) | 25.9 / 15.1 · loops 0 | 25.9 / 15.4 · loops 0 |
| stress/short (40) | 63.9 / 46.3 · loops 0 | 22.2 / 26.0 · loops 0 |
| stress/mixed (20) | 64.2 / 63.4 · loops 1 | 20.4 / 14.3 · loops 0 |

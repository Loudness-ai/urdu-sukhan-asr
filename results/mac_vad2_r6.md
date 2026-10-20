# mac_vad2_r6

whisper.cpp, beam 5 with temperature fallback, no timestamps (as the app runs). Each cell: word / letter error rate, loops, invented text on no-speech clips.

| Test set | r6c2775-q8_0 |
|---|---:|
| stress/mixed (20) | 41.8 / 38.6 · loops 0 |
| stress/noise (40) | loops 0 · invented 3/40 |
| stress/silence (30) | loops 0 · invented 0/30 |
| stress/quiet (40) | 23.9 / 17.1 · loops 0 |
| stress/short (40) | 23.6 / 26.8 · loops 0 |

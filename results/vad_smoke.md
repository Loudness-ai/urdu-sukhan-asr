# vad_smoke

whisper.cpp, beam 5 with temperature fallback, no timestamps (as the app runs). Each cell: word / letter error rate, loops, invented text on no-speech clips.

| Test set | r6c2775-q8_0 |
|---|---:|
| stress/silence (4) | loops 0 · invented 0/4 |
| stress/quiet (3) | 0.0 / 0.0 · loops 0 |

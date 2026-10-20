# Development log

Every problem met while building Urdu Sukhan ASR v1, with its cause and what was done about it, followed by the order
in which the work was done. The method and results are in the technical report
[LA-RB-2026-004](https://loudness.ai/research/urdu-speech-recognition-whisper/); this log keeps the
engineering details the report leaves out. “Laptop” is the Apple M4 MacBook Air used for rounds 1–3; “GPU server” is
the rented RTX A4000 machine; “cloud server” is the 12-core AMD machine used for CPU tests.

## Problems and fixes

### Setup, tooling and the laptop

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 1 | Model download stalled at 0 bytes | Hugging Face client download hung | Plain download with resume; no effect |
| 2 | Broken Urdu letters in drafts | whisper.cpp splits multi-byte letters across tokens | Bytes regrouped across tokens; drafts correct |
| 3 | Proofreading text box rendered empty | Size computed while the pane was hidden | Width guard and resize observer |
| 4 | A wait-then-score job would wait forever | Process search matched its own command line | Waits on process IDs |
| 5 | Background commands ran in the wrong folder | A directory change inside a command | Absolute paths throughout |
| 6 | Laptop disk nearly full (3.1 GB free) | Caches, model files, other sessions’ files | Caches cleared (5.5 GB); the large-v3 baseline was never finished |
| 7 | Round 1 stopped by its own disk guard | Brief disk spike while caching encoder outputs | Guard limited to caching; restart used 218 clips |
| 8 | Watcher reported the old log after a restart | It followed the deleted file | Re-pointed |
| 9 | Whisper large-v3 baseline killed twice | GPU contention, then disk | Not rerun; no large-v3 result exists |
| 10 | A word gained a growing row of tanween marks | House-style rule not idempotent | Rule applied once, repeated marks collapsed, all rules checked |
| 11 | A clip marked reviewed without being seen | Race between marking and new saves | Only clips listed in the review batch are marked |
| 12 | Laptop GPU deadlocked | Training (MPS) and whisper.cpp (Metal) waited on each other | One GPU job at a time; round-1 q5_0/q4_0 test never finished |
| 13 | Round 2 dropped the rest of clips at long pauses | Trained without timestamps, decoded with them | Decode without timestamps; from round 3, 30% timestamp labels; 4.8% to 3.2% |
| 14 | Laptop slowed by 5–9 GB of swap | Other applications | Rounds ran slower; live page about 5 s per phrase |
| 15 | Shorter audio window gave nonsense | Whisper is trained on full 30 s windows | Kept full windows; led to the untested audio-context proposal |
| 16 | In-app browser blocked the microphone | Browser policy | Live pages opened in Chrome, later over https |
| 17 | First drafts mixed English letters into Urdu words | Base model behaviour | Rules; English loanwords written in Urdu script in the books |
| 18 | First narrator drops the aspirated h | Speaker’s pronunciation | Meant word written; rounds 1–3 learned the voice |
| 19 | Round 1 flattered on its own test clips | Those clips were proofread from round-1 drafts | Stated in reports; not quantified |
| 20 | Test sets changed between rounds 1 and 4 | Growing data | Early numbers not comparable with later ones |
| 21 | Early checks capped outputs at 200 tokens | Default setting | 400 from round 4; not re-measured |

### GPU server, network and access

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 22 | PyTorch did not match the server’s driver | Default build targets newer CUDA | CUDA 12.6 build installed |
| 23 | Uploads failed over a phone hotspot | About 1 Mbps with drops | Batched copies with retries; only needed clips sent |
| 24 | Out of memory without gradient checkpointing | 15.6 GB for LoRA on encoder and decoder | Checkpointing kept (5 GB, 6.8 s per step) |
| 25 | Server login failed at first | Server still provisioning; key added to the wrong user | SSH key added for the server user |
| 26 | Parquet library missing on the laptop | Not installed | Importer run on the server |
| 27 | UAT transcription kept 43 of 600 clips | Confidence computed on misaligned scores | Confidence from per-step maxima; 99% kept |
| 28 | Round-5 smoke test out of memory | Transcription job held 9 GB of the GPU | Transcription paused at 7,019 clips and finished afterwards |
| 29 | Status connections dropped | Unstable network | Short commands, timeouts, detached jobs with logs |
| 30 | Session lost its connection after round 5 | API connection failure | Server idle until the session was resumed |
| 31 | Export to whisper.cpp failed | Output folder did not exist | Folder created by the script |
| 32 | Two copies of a scoring job ran | Launched twice | Extra copy killed; “exactly one copy” checked before reporting |
| 33 | Common Voice test file had duplicate rows | 313 rows, 260 unique clips | Unique file used; earlier Common Voice numbers not comparable |
| 34 | Desktop session restart killed five jobs | Application restart | Jobs relaunched detached |
| 35 | Live page slow and wrong on Urdu after English | Language forced to Urdu; context carried over | Auto language, no carried context, loop retry; 55 s to 17 s |
| 36 | R4+R5 average looped on very quiet clips | Quiet speakers | Temperature fallback removed the loops in whisper.cpp |
| 37 | MUSAN mirror had an invalid certificate | Mirror misconfigured | Not bypassed; another mirror used; noise data arrived late |
| 38 | Server unreachable | Server suspended for credit, first misdiagnosed | Credit added, server rebooted, downloads relaunched |
| 39 | Stopping a job killed our own connection | Kill pattern matched the connection | Bracket patterns; never stop and launch in one command |
| 40 | GPU left idle | Finished preparation went unnoticed | Training started when the author asked for an update |
| 41 | English stress set crashed the scorer | Language field “auto:en” | Generation and scoring languages separated |
| 42 | Code-switch clips dropped by cleaning | “Not Urdu script” rule | Code-switch clips exempted |
| 43 | IndicVoices download read only the first shards | Default download order | Shuffled download across shards; UrduSpeech resumed |
| 44 | Wrong relative paths in exported manifests | Export through a temporary folder | Paths rebuilt |
| 45 | Launch script copy failed silently | Dropped connection | Copy, verify, then launch |

### Data

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 46 | Inconsistent spelling and punctuation | Several accepted Urdu spellings | House style and punctuation removal on all text |
| 47 | Rules “fixed” real words | Single-word rules | Phrase-only rules for real words, each checked against all drafts |
| 48 | Meaning-based edits would teach the model to correct speakers | Review policy | Policy changed: transcribe what is said |
| 49 | A chapter title fell between two clips | Cutting at pauses | Fixed by hand |
| 50 | Common Voice clips whose audio ends before their text | Source data | Letters-per-second rule (27 removed) |
| 51 | FLEURS audio is very faint | Source recordings (median −45 dBFS) | See issue 58 |
| 52 | Stress-test MUSAN files used in round-6 augmentation | Augmentation read every MUSAN file | Round 6’s noise and music scores slightly flattering; student excludes them |
| 53 | Slowed-down long clips lose their last words | Speed augmentation near 30 s | Teaches invented endings; capped in the student trainer |

### Evaluation

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 54 | Loop rule changed three times | First rules counted real repetition as loops | Final rule compares with the reference; old counts too high (11 to 1) |
| 55 | Two-word rule hid one-word answers on silence | Definition of invented text | “Any word” counted; a “music 0/30” claim was corrected |
| 56 | Mixed stress clips have near-silent English halves | Raw-level FLEURS English | Voice detector rightly drops them; a test-set flaw |
| 57 | Peak normalisation worsened quiet speech | Not investigated | 23.5% to 53.6% for the released model |
| 58 | Round 6 learned “faint hiss is speech” | Faint FLEURS, very-quiet augmentation, echo filter level loss | Levelling fix used in the student; not ablated |
| 59 | Two whisper.cpp jobs deleted each other’s outputs | Outputs written next to shared inputs | False student collapse; per-job copies; reruns valid |
| 60 | Speed compared under unequal load | Other jobs running | Speed measured again on idle machines |

### Model behaviour

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 61 | One word on every silence clip | Blank suppression; empty labels encoded as a space | With suppression off, nothing on 100 clips, speech unchanged |
| 62 | Rounds 5 and 6 (final) collapsed on quiet audio | See the report’s sections on the training rounds and on silence | Averaging (round 5), checkpoint choice (round 6) |
| 63 | Runaway loops on odd quiet clips | Same | Released checkpoint writes a short wrong word instead |
| 64 | Names in the second audiobook | Names rare in training | 11 of 23 right; a name prompt gives 17; the author’s own name misspelt by q4_0 |
| 65 | English now in English letters | Round 6’s code-switch training | Preferred by the author; differs from the book house style |
| 66 | Voice detector hurts the mixed set | Issue 56 | Minimum pause 2 s and padding 300 ms on the test page |
| 67 | Student problems (hooks, memory, levels, silence) | See the report’s compression section | Fixed by restarts; student not released |

### Deployment and speed

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 68 | GPU server’s CPU a poor stand-in | 2012 Xeon without AVX2 | Tests moved to the cloud server |
| 69 | Auto language detection doubles encoder work | whisper.cpp encodes once to detect, again to transcribe | Language setting on the test page |
| 70 | q5_0 slowest on x86 | No optimised kernels for q5_0 there | Use q4_0 on x86 servers |
| 71 | 8 and 12 threads equally fast when the cloud server was quiet; 12 threads three times slower while its other services ran | Shared cores; threads waiting on each other | About 8 threads per job on 12 shared cores; jobs side by side |
| 72 | faster-whisper not the expected 2–4 times faster (1.5–1.8 times real time) | int8 kernels tuned for Intel; converter and audio-reader bugs on the way | Not adopted on this server; whisper.cpp on the same file not run |
| 73 | Every clip pays for a 30 s window | whisper.cpp design | Audio-context fine-tune proposed (3–4 GPU-hours), not run |
| 74 | Full-model Core ML encoder imprecise | fp16 over 32 layers (cosine 0.984) | Not released |
| 75 | Hugging Face upload refused | Read-only token | Fine-grained write token added by the author; upload done |
| 76 | First code commit included the paper and a private name | Draft files committed | Removed; history squashed before publication |
| 77 | LaTeX build error | Package order | Right-to-left package loaded last |
| 78 | Test page needed https for the microphone | Browser policy | Self-signed https; unprivileged user |
| 79 | Commands run by the author failed | Placeholder file name; wrong folder | Terminal opened in the project folder |
| 80 | Laptop backup check compared wrong folders | Script error | Redone; 2,830 files verified |

### Process

| # | Problem | Cause | Fix and effect |
|---:|---|---|---|
| 81 | Status reported before checking | Haste | Smoke test first, one copy confirmed, proactive reports |
| 82 | Side sessions used the author’s quota | Parallel planning sessions | Detached on request |
| 83 | GPU server shut down with the only full-precision weights | Server discarded at the end of the work | Released model survives in whisper.cpp and CTranslate2; Transformers weights rebuilt from q8_0 and checked against both |

## Order of work

1. Goals set: better Urdu, lower memory, faster on CPUs and iPhones.
2. whisper.cpp, normaliser, proofreading tool and scorer built; base-model baseline; house style adopted.
3. The two audiobooks cut into 1,037 clips and drafted; proofreading and the five-clip review loop start.
4. Round 1 on the laptop (a first attempt stopped by the disk guard).
5. Round 2; timestamp bug found and fixed in decoding.
6. Round 3 with timestamp labels; live test page on the laptop.
7. GPU server rented and set up; nzg73 and UAT (audio only) chosen; round 4.
8. Round 5, the first full fine-tune.
9. Round 5’s loops on quiet speakers found; R4+R5 averages built; the 50/50 average becomes the live default.
10. Round-6 plan: stress sets, cleaning, no-speech and code-switch clips, IndicVoices and UrduSpeech.
11. Round 6 trains (59,165 clips, 3,697 steps).
12. Checkpoints and three averages scored; step 2,775 chosen.
13. Compression pilots (16 and 8 encoder layers).
14. Student main run with two restarts; one-word answers traced to blank suppression.
15. Student extension; S1 chosen at step 6,600.
16. CPU test page on the cloud server; the author ranks the released model first; faster-whisper tested.
17. Release prepared as Urdu Sukhan ASR v1; the technical report drafted.
18. GPU server shut down; Transformers weights rebuilt from q8_0 (issue 83).
19. Side-by-side examples against the base model; publication on Hugging Face, GitHub and loudness.ai.

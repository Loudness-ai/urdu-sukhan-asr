# whisper-lite

Make Whisper better at Urdu (and other languages), and small and fast enough
for ordinary computers and iPhones.

The idea: public Urdu speech data is small (FLEURS has about 7 hours of
training audio, Common Voice mostly read sentences), so the biggest gain comes
from **proofread transcripts of real audio like yours**. This repo is the loop
that makes that cheap:

```
your recordings ──► cut at pauses ──► Whisper draft ──► you correct ──► training data
      ▲                                                                    │
      └──────────── better model writes the next drafts ◄── fine-tune ◄────┘
```

## Setup (once)

```bash
brew install whisper-cpp uv ffmpeg
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## 1. Transcribe your audio

```bash
.venv/bin/python scripts/transcribe.py ~/Desktop/urdu_audio --project batch1
```

Takes files or folders (mp3, m4a, wav, mp4, …). Each recording is cut at
pauses into clips of up to 25 seconds, and each clip gets a draft from
Whisper large-v3-turbo running on the Mac's GPU, with a confidence score for
every word. Run it again with new files to add them to the same project.

## 2. Proofread

```bash
.venv/bin/python scripts/proofread.py batch1
```

Open http://localhost:8765. Red words are the ones the model was unsure about;
click any word to hear it. Everything is saved to `projects/batch1/items.jsonl`
as soon as you save a clip.

| Key | Does |
|---|---|
| Tab | play / pause |
| Shift+Tab | replay the clip from the start |
| Ctrl+Enter | save as correct, go to next |
| Ctrl+Shift+Enter | mark unusable, go to next |
| Ctrl+[ / Ctrl+] | back / skip |

The header shows **how wrong the model's drafts were on the clips you've
corrected**. That number is the honest measure of the model on your kind of
audio, and it should drop after each training round.

### Proofreading rules

Training data has to be consistent, or the model learns to be inconsistent.

1. Write what is said, in Urdu script. No punctuation needed: for now it is
   removed from all training text (ours and public), so the model learns to
   write plain text. Switch back with `KEEP_PUNCTUATION` in `wlite/text.py`.
2. English words spoken inside Urdu: write them the way you always will.
   Pick Urdu script (کمپیوٹر) or English letters (computer) once and stick to it.
3. Numbers: always digits, or always words. Pick one.
4. Leave out filler sounds (اں، ہمم) and coughs.
5. If you can't make out a word after two listens, or there is music,
   cross-talk or another language, mark the clip **Unusable** instead of guessing.

### Urdu house style (applied automatically)

These spellings are enforced in every training sentence (ours and public
datasets), in the drafts you proofread, and on both sides before scoring.
Both forms are correct Urdu; using one keeps the model consistent.
Defined in `URDU_STYLE_WORDS` in [wlite/text.py](wlite/text.py).

| Written as | Instead of |
|---|---|
| لیے، کے لیے، اس لیے | لئے، لیئے، کیلئے، کیلیے، اسلئے |
| کیے، دیے، چاہیے | کئے، دیئے، چاہئے |
| انہیں، انہوں، جنہیں، تمہیں، تمہارا | انھیں، انھوں، جنھیں، تمھیں، تمھارا |
| کر کے، ہو گئی، جا سکتا، کر دیا، آ رہا… | کرکے، ہوگئی، جاسکتا، کردیا، آرہا… |
| دراصل | در اصل |
| یورپ | یوروپ |

Left alone on purpose: future forms (ہوگا / ہو گا — no agreed form) and words
that only look like a verb plus helper (ہولی، جالی).

### Fixes learned from your corrections

The model repeats its mistakes (in the novel it writes مصبت/مذبت/مضبط for
مثبت dozens of times). `projects/<name>/corrections.tsv` lists fixes that are
right in every context; the page applies them to clips you haven't proofread
yet and shows those words in **blue** (hover to see what the model wrote).
Spoken years (انیس سو چھیاسی → ۱۹۸۶) and Western digits (→ Urdu digits) are
converted automatically.

```bash
.venv/bin/python scripts/learn_corrections.py batch1
```

lists new fixes you keep making. Add only the safe ones to the file (not real
words that are only sometimes wrong, like تھی/تھیں or ہاتھ/ہار); the page picks
them up within 30 seconds.

## 3. Export the corrected clips

```bash
.venv/bin/python scripts/export_dataset.py batch1
```

Writes `data/proofread/batch1/{train,test}.jsonl`. About 10% is held out for
testing, chosen **by recording**, so the test never contains a speaker or
session the model trained on.

## Scoring models

```bash
.venv/bin/python scripts/get_fleurs.py ur_pk --splits test
.venv/bin/python scripts/evaluate.py data/fleurs/ur_pk/test.jsonl \
    --models small-q8_0 large-v3-turbo-q8_0 --name fleurs_ur_baseline
```

Reports word error rate (WER), character error rate (CER) and speed in
`results/<name>.md`. Before scoring, both texts are normalized: Arabic vs
Urdu letter forms (ي/ی، ك/ک، ه/ہ), Urdu vs Western digits, diacritics and
punctuation are made consistent, so only real mistakes count.

`get_fleurs.py` works for any FLEURS language (`hi_in`, `pa_in`, `fa_ir`,
`ar_eg`, `en_us`, …), which is how other languages get mixed into training.

## Layout

```
wlite/           shared code: text normalization, audio, whisper.cpp runner, projects
scripts/         command-line steps above
proofread/       the proofreading page
projects/<name>  your clips and corrections (keep these safe; they are the valuable part)
data/            downloaded datasets and exported manifests
models/          downloaded and trained models
results/         scores
```

## Coming next

- Fine-tune a large "teacher" on your corrected clips + public Urdu data.
- Distill it into a small "student" with fewer decoder layers (Urdu needs many
  tokens per word, so fewer decoder layers is where Urdu speed comes from).
- Export: int8 for computers (faster-whisper), 4-bit Core ML for iPhone
  (WhisperKit), and a whisper.cpp file that runs on both.

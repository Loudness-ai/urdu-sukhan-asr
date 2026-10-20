"""Distil a trained Whisper teacher into a smaller student (made by scripts/student_init.py). CUDA only.

  python scripts/train_student.py --teacher models/runs/novel-r6/ckpt-2775 --student models/students/init-e16d4 \
      --train data/r6/cv.jsonl data/r6/indicvoices.jsonl ... --nospeech 0.12 \
      --noise-dir data/musan/musan/noise data/musan/musan/music --out models/runs/student-e16d4 --save-every 300

Each step the teacher (frozen, bf16) hears the same augmented audio as the student, and the student learns from
  - the transcript (cross-entropy),
  - the teacher's next-token probabilities on that transcript (KL divergence, temperature --kd-temp),
  - the teacher's encoder output, and its layers that the student's kept layers came from (MSE).
No-speech clips (empty text) are trained on the transcript only: the answer is "nothing", not whatever
the teacher would have guessed. So are the proofread book clips (--human-only): the teacher misspells
Book 2's names (عمیر -> امیر, عزیر -> اوزیر, تانیا -> دانیا) and drops the narrator's ھ, so on those clips the
student learns only the human text. They are never packed with other clips and never held out as dev.

Silence and noise, after rounds 5 and 6 learned to write text on faint hiss:
  - every speech clip is first brought to a normal level (about -20 dB RMS, peaks capped): some FLEURS
    recordings are -45 to -63 dB, as faint as the hiss in the silence test, and taught "faint = words";
  - training levels then stay in -28 to -12 dB, with a quiet share at -40 to -28 dB (never fainter; the FLEURS
    test itself sits at -35 to -42 dB, the silence test's hiss at -52 to -59 dB), and
    quiet speech's background noise stays 15-35 dB below it;
  - --nospeech adds generated no-speech clips (digital silence, hiss, hum, real MUSAN noise/music at
    every level from -90 dB to full scale), a share of every epoch, new audio each time;
  - speech clips are sometimes padded with faint room noise before/after, so the model learns to stop
    where the words stop;
  - Urdu + English code-switch windows are built on the fly from separately levelled clips (the stored
    data/r6/codeswitch.jsonl joins clips at their raw levels, so its English half can be near-silent).
MUSAN files held out for the stress test (md5(name) % 10 == 0) are never used.

Every --save-every steps a checkpoint is saved and quickly checked on held-out clips (greedy decoding):
invented text on noise/music/silence stress clips, loops, and WER on a dev slice held out of --train.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train as base  # noqa: E402  (manifest reading, MUSAN loader, collate)
from eval_hf import runaway  # noqa: E402
from wlite.audio import load_audio  # noqa: E402
from wlite.text import clean_label, word_char_errors  # noqa: E402

SR = 16000


def held_out(path):
    return int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 10 == 0


def db(x):
    return 10 ** (x / 20)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def set_level(audio, target_db, peak=0.99):
    """Scale to a target RMS level (dB re full scale), without letting peaks pass `peak`."""
    if not len(audio):
        return audio
    g = db(target_db) / rms(audio)
    g = min(g, peak / (float(np.abs(audio).max()) + 1e-9))
    return (audio * g).astype(np.float32)


def colored_noise(n, rng):
    """White, pink-ish or brown-ish noise with unit RMS."""
    x = rng.standard_normal(n).astype(np.float32)
    kind = rng.random()
    if kind < 0.33:
        x = np.cumsum(x)
        x -= np.convolve(x, np.ones(400) / 400, "same")
    elif kind < 0.66:
        k = rng.integers(2, 12)
        x = np.convolve(x, np.ones(k) / k, "same")
    return (x / rms(x)).astype(np.float32)


def room(audio, rng):
    n = int(SR * rng.uniform(0.1, 0.4))
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.linspace(0, rng.uniform(4, 8), n))
    ir[0] = 1.0
    return np.convolve(audio, ir / np.abs(ir).sum() * 2)[: len(audio)].astype(np.float32)


def phone(audio):
    import soxr
    x = soxr.resample(soxr.resample(audio, SR, 8000), 8000, SR).astype(np.float32)[: len(audio)]
    return np.clip(x * 1.5, -0.5, 0.5)


NOSPEECH_FILES = []  # MUSAN noise + music without singing (training part): backgrounds for no-speech clips


def background(n, rng, files=None):
    """n samples of a real recording (MUSAN, training part only) or generated noise, unit RMS."""
    import soundfile as sf
    import soxr

    files = base.NOISE_FILES if files is None else files
    if not files or rng.random() >= 0.7:
        return colored_noise(n, rng)
    path = files[rng.integers(len(files))]
    info = sf.info(path)
    need = int(n * info.samplerate / SR) + 1
    x, sr = sf.read(path, start=int(rng.integers(0, max(1, info.frames - need))), frames=need, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    if len(x) < n:
        x = np.tile(x, n // max(len(x), 1) + 1)
    x = x[:n].astype(np.float32)
    return x / rms(x)


def no_vocals(files):
    """Drop MUSAN music tracks annotated as having vocals (a song is not "nothing said")."""
    vocals = {}
    for ann in {Path(f).parent / "ANNOTATIONS" for f in files}:
        if ann.exists():
            for line in ann.read_text(errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    vocals[parts[0]] = parts[2].upper() == "Y"
    return [f for f in files if "/music/" not in f or not vocals.get(Path(f).stem, True)]


def augment_speech(audio, rng):
    """Speed, loudness, room, phone line, background noise, faint padding. Returns audio, speed, lead pad (s)."""
    import soxr

    speed = rng.uniform(0.9, 1.1) if rng.random() < 0.5 else 1.0
    speed = max(speed, len(audio) / (29.5 * SR))  # never stretch words past the 30 s window
    if speed != 1.0:
        audio = soxr.resample(audio, SR * speed, SR).astype(np.float32)
    if rng.random() < 0.3:
        audio = room(audio, rng)
    if rng.random() < 0.15:
        audio = phone(audio)
    # level last: the room filter alone lowers the level by ~20 dB (its response is scaled by its sum, as in
    # train.py), which is how round 6 still saw faint speech with words
    quiet = rng.random() < 0.15  # far from the mic, but never as faint as the no-speech hiss
    audio = set_level(audio, rng.uniform(-40, -28) if quiet else rng.uniform(-28, -12))
    lead = 0.0
    level = rms(audio)
    room_left = 29.8 - len(audio) / SR
    if rng.random() < 0.3 and room_left > 0.3:  # faint room noise before/after the words
        lead, tail = rng.uniform(0, min(2.0, room_left / 2)), rng.uniform(0, min(3.0, room_left / 2))
        room_tone = colored_noise(int((lead + tail) * SR), rng) * level * db(-rng.uniform(25, 45))
        cut = int(lead * SR)
        audio = np.concatenate([room_tone[:cut], audio, room_tone[cut:]])[: 30 * SR]
    if rng.random() < 0.6:  # background noise; quiet speech always stays well above it
        snr = rng.uniform(15, 35) if quiet else rng.uniform(0, 30)
        audio = audio + background(len(audio), rng) * level * db(-snr)
    return np.clip(audio, -1, 1).astype(np.float32), speed, lead


def nospeech_audio(rng):
    """A clip with nothing said: silence, hiss, hum, or real noise/music, at any level."""
    dur = float(np.exp(rng.uniform(np.log(0.5), np.log(30))))
    n = int(dur * SR)
    kind = rng.random()
    if kind < 0.15:  # digital silence, or dither just above it
        x = np.zeros(n, np.float32) if rng.random() < 0.5 else colored_noise(n, rng) * db(rng.uniform(-100, -80))
    elif kind < 0.5:  # microphone / room hiss at the levels quiet speech also has
        x = colored_noise(n, rng) * db(rng.uniform(-90, -40))
        if rng.random() < 0.3:
            t = np.arange(n) / SR
            f = rng.choice([50, 60])
            hum = sum(np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.3)) / k for k in range(1, 4)).astype(np.float32)
            x += hum / rms(hum) * db(rng.uniform(-70, -35))
    else:  # real background recordings, quiet to loud
        x = background(n, rng, NOSPEECH_FILES) * db(rng.uniform(-60, -10))
        if rng.random() < 0.3:
            x += colored_noise(n, rng) * db(rng.uniform(-80, -50))
    if rng.random() < 0.2:
        x = room(x, rng)
    if rng.random() < 0.1:
        x = phone(x)
    if rng.random() < 0.2:  # a stretch of digital silence around it, as when a recording starts/stops
        pad = np.zeros(int(rng.uniform(0.2, 3) * SR), np.float32)
        x = np.concatenate([pad, x, pad] if rng.random() < 0.5 else [x, pad])[: 30 * SR]
    return np.clip(x, -1, 1).astype(np.float32)


NOSPEECH = {"nospeech": True, "language": "ur", "text": ""}


def windows_of(rows, pack, n_nospeech, seed, codeswitch=0.0):
    """Group clips into 30 s training windows. A share `pack` of windows holds 2-4 short clips of one
    language back to back (up to 25 s): much more speech per GPU-second, and practice at longer audio.
    The rest hold one clip, as in the app; no-speech clips are always alone."""
    rng = random.Random(seed)
    speech = [r for r in rows if r["text"]]
    order = list(range(len(speech)))
    rng.shuffle(order)
    out = []
    while order:
        r = speech[order.pop()]
        dur = float(r.get("duration") or 0)
        if rng.random() >= pack or not 0 < dur <= 18 or r.get("kd") is False:
            out.append([r])
            continue
        want, win, total = rng.randint(2, 4), [r], dur
        j = len(order) - 1
        while len(win) < want and j >= max(0, len(order) - 30):
            c = speech[order[j]]
            d = float(c.get("duration") or 0)
            if c["language"] == r["language"] and 0 < d and total + d + 1.0 <= 25 and c.get("kd") is not False:
                win.append(c)
                total += d + 1.0
                order.pop(j)
            j -= 1
        out.append(win)
    ur = [r for r in speech if r["language"] == "ur" and r.get("kd") is not False and 0 < float(r.get("duration") or 0) <= 15]
    en = [r for r in speech if r["language"] == "en" and 0 < float(r.get("duration") or 0) <= 12]
    for _ in range(int(codeswitch * len(out)) if ur and en else 0):  # Urdu then English, sometimes the other way
        a, b = rng.choice(ur), rng.choice(en)
        out.append([a, b] if rng.random() < 0.7 else [b, a])
    out += [[r] for r in rows if not r["text"]]
    out += [[NOSPEECH]] * n_nospeech
    return out


class StudentClips(torch.utils.data.Dataset):
    """Training windows: one manifest clip, 2-4 packed clips, or a generated no-speech clip."""

    def __init__(self, rows, processor, timestamp_frac, augment, n_nospeech, pack=0.0, seed=0, codeswitch=0.0,
                 forced_words=(), forced_frac=0.0):
        self.windows = windows_of(rows, pack, n_nospeech, seed, codeswitch)
        self.fe, self.tok = processor.feature_extractor, processor.tokenizer
        self.forced_words, self.forced_frac = list(forced_words), forced_frac
        self.timestamp_frac, self.augment = timestamp_frac, augment
        self.ts_begin = self.tok.convert_tokens_to_ids("<|notimestamps|>") + 1
        assert self.tok.convert_tokens_to_ids("<|0.00|>") == self.ts_begin

    def __len__(self):
        return len(self.windows)

    def labels(self, r, speed=1.0, lead=0.0):
        lang = r["language"]
        task = self.tok.convert_tokens_to_ids([f"<|{lang}|>", "<|transcribe|>"])
        if not r["text"]:
            return task + self.tok.convert_tokens_to_ids(["<|notimestamps|>"]) + [self.tok.eos_token_id]
        ids = self.tok.encode(" " + r["text"], add_special_tokens=False)[: base.MAX_LABEL - 5]
        if self.timestamp_frac and "speech_end" in r and random.random() < self.timestamp_frac:
            ts = lambda t: self.ts_begin + min(1500, max(0, round(t / 0.02)))
            return (task + [ts(r["speech_start"] / speed + lead)] + ids + [ts(r["speech_end"] / speed + lead)]
                    + [self.tok.eos_token_id])
        return task + self.tok.convert_tokens_to_ids(["<|notimestamps|>"]) + ids + [self.tok.eos_token_id]

    def __getitem__(self, i):
        win = self.windows[i]
        rng = np.random.default_rng(random.getrandbits(32))
        speed, lead = 1.0, 0.0
        if len(win) > 1:  # packed: clips back to back with short pauses, then augmented as one recording
            parts = []
            for k, c in enumerate(win):
                if k:
                    parts.append(np.zeros(int(rng.uniform(0.2, 1.0) * SR), np.float32))
                parts.append(set_level(load_audio(c["audio_path"]), rng.uniform(-24, -16)))  # each at a normal level
            audio = np.concatenate(parts)[: 30 * SR]
            if self.augment and random.random() < self.augment:
                audio, speed, lead = augment_speech(audio, rng)
            lang = "ur" if any(c["language"] == "ur" for c in win) else win[0]["language"]
            r = {"language": lang, "text": " ".join(c["text"] for c in win)}
            feats = self.fe(audio[: 30 * SR], sampling_rate=SR, return_tensors="np").input_features[0]
            return feats, self.labels(r), True, []
        r = win[0]
        if r.get("nospeech"):
            audio = nospeech_audio(rng)
        else:
            audio = load_audio(r["audio_path"])
            if not r["text"]:  # recorded no-speech clips (data/r6/nospeech.jsonl): natural levels, small changes
                if self.augment and random.random() < self.augment:
                    audio = np.clip(audio * db(rng.uniform(-30, 6)), -1, 1).astype(np.float32)
            elif self.augment and random.random() < self.augment:
                audio, speed, lead = augment_speech(audio, rng)
            else:
                audio = set_level(audio, rng.uniform(-23, -17))
        feats = self.fe(audio[: 30 * SR], sampling_rate=SR, return_tensors="np").input_features[0]
        forced = []
        if not r["text"] and self.forced_words and random.random() < self.forced_frac:
            # whisper.cpp and HF forbid end-of-text as the first token, so on no-speech the decoder is made to say
            # some word first; train the answer to that situation too: after a forced word, stop at once.
            forced = self.forced_words[int(rng.integers(len(self.forced_words)))]
        return feats, self.labels(r, speed, lead), r.get("kd", True) and bool(r["text"]), forced


SOT, EOT = 50258, 50257  # <|startoftranscript|>, <|endoftext|> in every multilingual Whisper vocab


def collate(batch):
    """Features, labels (-100 = not scored), decoder inputs, teacher-loss flags. A forced word sits in the decoder
    input after <|notimestamps|> but is not scored; only the end-of-text after it is."""
    feats = torch.from_numpy(np.stack([b[0] for b in batch]))
    labs, decs = [], []
    for _, lab, _, forced in batch:
        if forced:
            decs.append([SOT] + lab[:3] + list(forced))
            labs.append(lab[:3] + [-100] * len(forced) + [EOT])
        else:
            decs.append([SOT] + lab[:-1])
            labs.append(lab)
    width = max(len(l) for l in labs)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    dec = torch.full((len(batch), width), EOT, dtype=torch.long)
    for i, (l, d) in enumerate(zip(labs, decs)):
        labels[i, : len(l)] = torch.tensor(l)
        dec[i, : len(d)] = torch.tensor(d)
    return feats, labels, dec, torch.tensor([b[2] for b in batch])


def layer_outputs(layers, store):
    """Forward hooks keeping each encoder layer's output (first forward only, not recomputation)."""
    hooks = []
    for j, layer in enumerate(layers):
        def hook(mod, inp, out, j=j):
            if j not in store:
                store[j] = out[0] if isinstance(out, tuple) else out
        hooks.append(layer.register_forward_hook(hook))
    return hooks


@torch.no_grad()
def quick_eval(model, processor, sets, device, batch=16):
    """Greedy decoding on a few held-out sets: WER, loops, invented text on no-speech clips."""
    model.eval()
    res = {}
    for name, rows in sets.items():
        hyps = []
        for i in range(0, len(rows), batch):
            part = rows[i : i + batch]
            audio = [load_audio(r["audio_path"])[: 30 * SR] for r in part]
            feats = processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
            lang = part[0]["language"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(input_features=feats.to(device), language=None if lang.startswith("auto") else lang,
                                     task="transcribe", max_new_tokens=300, num_beams=1, use_cache=True)
            hyps += [h.strip() for h in processor.tokenizer.batch_decode(out, skip_special_tokens=True)]
        lang = rows[0]["language"]
        lang = lang.split(":", 1)[1] if ":" in lang else ("ur" if lang == "auto" else lang)
        refs = [clean_label(r["text"], lang) for r in rows]
        speech = [k for k, r in enumerate(refs) if r]
        cell = {"loops": sum(runaway(h, r) for h, r in zip(hyps, refs) if r)}
        if speech:
            cell["wer"], cell["cer"] = word_char_errors([refs[k] for k in speech], [hyps[k] for k in speech], lang)
        silent = [k for k, r in enumerate(refs) if not r]
        if silent:
            cell["invented"] = sum(len(hyps[k].split()) >= 2 for k in silent)
            cell["one_word"] = sum(len(hyps[k].split()) == 1 for k in silent)  # not counted as invented, still wrong
            cell["loops"] += sum(len(hyps[k].split()) > 12 for k in silent)
        res[name] = cell
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nospeech", type=float, default=0.12, help="generated no-speech clips, as a share of the speech windows")
    ap.add_argument("--pack", type=float, default=0.5, help="share of speech windows holding 2-4 clips back to back")
    ap.add_argument("--codeswitch", type=float, default=0.0,
                    help="extra Urdu+English windows built on the fly, as a share of the speech windows (use instead of codeswitch.jsonl)")
    ap.add_argument("--human-only", nargs="*", default=["book1", "book2"],
                    help="manifest names (file stems) trained on their human text only: no teacher loss, no packing, no dev hold-out")
    ap.add_argument("--holdout", type=int, default=60, help="1 in N clips of each --train manifest kept out as dev")
    ap.add_argument("--dev-max", type=int, default=120, help="dev clips per manifest used by the checkpoint check")
    ap.add_argument("--quick-sets", nargs="*", default=["data/stress/noise.jsonl", "data/stress/music.jsonl",
                                                         "data/stress/silence.jsonl", "data/stress/quiet.jsonl:20",
                                                         "data/stress/short.jsonl:20"])
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--nospeech-weight", type=float, default=1.0, help="weight of the end-of-text token of no-speech windows in the transcript loss")
    ap.add_argument("--forced-word", type=float, default=0.0,
                    help="share of no-speech windows trained as 'a word was forced first, now stop' (see collate)")
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--enc", type=float, default=1.0, help="weight of the encoder-output match")
    ap.add_argument("--enc-layers", type=float, default=0.5, help="weight of the per-layer match (layer-normalised)")
    ap.add_argument("--augment", type=float, default=0.5)
    ap.add_argument("--spec-augment", action="store_true")
    ap.add_argument("--noise-dir", nargs="*", default=[])
    ap.add_argument("--timestamp-frac", type=float, default=0.2)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for d in args.noise_dir:
        files = [str(p) for p in Path(d).rglob("*") if p.suffix.lower() in (".wav", ".flac")]
        base.NOISE_FILES.extend(p for p in files if not held_out(p))
    NOSPEECH_FILES.extend(no_vocals(base.NOISE_FILES))
    print(f"background recordings (stress-test files excluded): {len(base.NOISE_FILES)}, "
          f"{len(NOSPEECH_FILES)} without singing for no-speech clips", flush=True)

    processor = WhisperProcessor.from_pretrained(args.teacher)
    teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher, dtype=torch.bfloat16).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = WhisperForConditionalGeneration.from_pretrained(args.student, dtype=torch.float32).to(device)
    student.generation_config.forced_decoder_ids = None
    student.config.use_cache = False
    if args.spec_augment:  # student only: the teacher hears the unmasked audio and gives the clean answer
        student.config.update(dict(apply_spec_augment=True, mask_time_prob=0.05, mask_time_length=10, mask_time_min_masks=2,
                                   mask_feature_prob=0.05, mask_feature_length=10, mask_feature_min_masks=0))
    else:
        student.config.apply_spec_augment = False
    if args.grad_ckpt:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    kept = json.loads((Path(args.student) / "student_init.json").read_text())["encoder_layers"]
    print(f"student {sum(p.numel() for p in student.parameters()) / 1e6:.0f}M, encoder layers from teacher {kept}", flush=True)

    train_rows, dev = [], {}
    for spec in args.train:
        rows, rep = base.read_manifest(spec)
        human = Path(spec.partition(":")[0]).stem in args.human_only
        for r in rows:
            r["kd"] = not human
        is_dev = [bool(args.holdout) and not human and bool(r["text"])
                  and int(hashlib.md5(r["audio_path"].encode()).hexdigest(), 16) % args.holdout == 0 for r in rows]
        keep = [r for r, d in zip(rows, is_dev) if not d]
        held = [r for r, d in zip(rows, is_dev) if d]
        train_rows += keep * (rep or 1)
        if held:
            dev[f"dev/{Path(spec.partition(':')[0]).stem}"] = held[: args.dev_max]
        print(f"train {spec}: {len(keep)} clips x{rep or 1}  (dev {len(held)}){'  human text only' if human else ''}", flush=True)
    quick = {}
    for spec in args.quick_sets:
        path, _, n = spec.partition(":")
        rows = [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]
        for r in rows:
            r["audio_path"] = str((Path(path).parent / r["audio"]).resolve())
        quick[f"stress/{Path(path).stem}"] = rows[: int(n)] if n else rows
    quick.update(dev)
    n_windows = len(windows_of(train_rows, args.pack, 0, args.seed, args.codeswitch))
    n_nospeech = int(args.nospeech * n_windows)
    forced_words = []
    if args.forced_word:  # the commonest words of the transcripts, as the decoder would pick them when forced
        from collections import Counter
        counts = {"ur": Counter(), "en": Counter()}
        for r in train_rows:
            if r["text"] and r["language"] in counts:
                counts[r["language"]].update(r["text"].split())
        words = [w for w, _ in counts["ur"].most_common(400)] + [w for w, _ in counts["en"].most_common(100)]
        forced_words = [processor.tokenizer.encode(" " + w, add_special_tokens=False) for w in words]
        print(f"forced-word no-speech windows: {args.forced_word:.0%}, from {len(forced_words)} common words", flush=True)
    dataset = StudentClips(train_rows, processor, args.timestamp_frac, args.augment, n_nospeech, args.pack, args.seed,
                           args.codeswitch, forced_words, args.forced_word)
    packed = sum(len(w) > 1 for w in dataset.windows)
    print(f"{len(train_rows)} manifest clips in {n_windows} windows ({packed} packed) + {n_nospeech} generated "
          f"no-speech clips per epoch", flush=True)
    (out / "dev_ids.json").write_text(json.dumps({k: [r["audio_path"] for r in v] for k, v in dev.items()}, indent=1))

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                         num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers else None)
    total = args.max_steps or math.ceil(len(loader) // args.accum * args.epochs)
    params = [p for p in student.parameters() if p.requires_grad]
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    warm = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, (total - s) / max(1, total - warm)))
    print(f"{total} steps of {args.batch * args.accum} clips", flush=True)

    t_store, s_store = {}, {}
    layer_hooks = []
    if args.enc_layers:
        layer_hooks += layer_outputs([teacher.model.encoder.layers[i] for i in kept], t_store)
        layer_hooks += layer_outputs(student.model.encoder.layers, s_store)
    enc_out = {}  # hooks must return None, or they replace the module's output

    def keep_enc(key):
        def hook(mod, inp, out):
            if key not in enc_out:
                enc_out[key] = out.last_hidden_state
        return hook
    teacher.model.encoder.register_forward_hook(keep_enc("t"))
    student.model.encoder.register_forward_hook(keep_enc("s"))

    def save(step, evaluate=True):
        ck = out / f"ckpt-{step}"
        student.save_pretrained(ck, state_dict={k: v.half() for k, v in student.state_dict().items()})
        processor.save_pretrained(ck)
        for f in ("student_init.json",):
            (ck / f).write_text((Path(args.student) / f).read_text())
        line = {"step": step}
        if evaluate:
            t = time.time()
            line.update(quick_eval(student, processor, quick, device))
            t_store.clear(); s_store.clear(); enc_out.clear()
            line["eval_s"] = round(time.time() - t)
        with open(out / "checks.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"saved {ck}  {json.dumps(line)}", flush=True)

    student.train()
    step, t0, run = 0, time.time(), {}
    T = args.kd_temp
    while step < total:
        for i, (feats, labels, dec, use_kd) in enumerate(loader):
            feats, labels = feats.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            dec, use_kd = dec.to(device, non_blocking=True), use_kd.to(device)
            t_store.clear(); s_store.clear(); enc_out.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    t_logits = teacher(input_features=feats, decoder_input_ids=dec, labels=labels).logits.float()
                s_out = student(input_features=feats, decoder_input_ids=dec, labels=labels)
            mask = labels != -100
            # no-speech and proofread book clips: transcript only
            kd_mask = mask & use_kd[:, None]
            s_logits = s_out.logits.float()
            if args.nospeech_weight != 1.0:
                # A no-speech window has one real decision, end-of-text right after <|notimestamps|>, against ~1,300
                # transcript tokens in a batch, so with a plain per-token mean it barely counts; weight that token.
                tok_ce = F.cross_entropy(s_logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
                stop = (mask.sum(1) <= 4)[:, None] & (labels == processor.tokenizer.eos_token_id)
                w = torch.where(stop, args.nospeech_weight, 1.0) * mask
                loss_ce = (tok_ce * w).sum() / w.sum()
            else:
                loss_ce = s_out.loss
            if kd_mask.any():
                lp_s = F.log_softmax(s_logits[kd_mask] / T, -1)
                p_t = F.softmax(t_logits[kd_mask] / T, -1)
                loss_kd = F.kl_div(lp_s, p_t, reduction="batchmean") * T * T
            else:
                loss_kd = s_logits.sum() * 0
            loss_enc = F.mse_loss(enc_out["s"].float(), enc_out["t"].float())
            loss_layers = torch.zeros((), device=device)
            if args.enc_layers:
                for j in range(len(kept)):
                    a, b = s_store[j].float(), t_store[j].float()
                    loss_layers = loss_layers + F.mse_loss(F.layer_norm(a, a.shape[-1:]), F.layer_norm(b, b.shape[-1:]))
                loss_layers = loss_layers / len(kept)
            loss = args.ce * loss_ce + args.kd * loss_kd + args.enc * loss_enc + args.enc_layers * loss_layers
            (loss / args.accum).backward()
            for k, v in (("ce", loss_ce), ("kd", loss_kd), ("enc", loss_enc), ("lay", loss_layers)):
                run.setdefault(k, []).append(float(v))
            if (i + 1) % args.accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                el = time.time() - t0
                parts = "  ".join(f"{k} {np.mean(v):.3f}" for k, v in run.items())
                print(f"step {step}/{total}  {parts}  gnorm {float(gn):.2f}  lr {sched.get_last_lr()[0]:.1e}  "
                      f"{el / step:.2f}s/step  eta {(total - step) * el / step / 60:.0f} min  "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                run = {}
            if args.save_every and step % args.save_every == 0 and step < total:
                save(step)
            if step >= total:
                break
    save(step)
    print("STUDENT_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

"""Fine-tune Whisper with LoRA on proofread clips (+ public data). Mac GPU (MPS), CUDA or CPU.

  python scripts/train.py --base models/hf/whisper-large-v3-turbo \
      --train data/proofread/novel/train.jsonl:4 data/fleurs/ur_pk/train.jsonl:1 \
      --eval data/proofread/novel/test.jsonl data/fleurs/ur_pk/test.jsonl:100 \
      --out models/runs/novel-r1 --decoder-only --epochs 2

':N' after a --train manifest repeats it N times (proofread clips count more than public
ones); ':~K' takes a random K clips from it (for big sets that would drown the rest). After an
--eval manifest ':N' scores only the first N clips.

--decoder-only freezes the encoder and adds LoRA adapters to the decoder: about 3 GB of
memory instead of 10+, roughly twice as fast, and it targets what proofreading fixes most
(spelling, names, house style, words the narrator under-pronounces).
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wlite.audio import load_audio  # noqa: E402
from wlite.text import clean_label, word_char_errors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_LABEL = 440  # Whisper's decoder holds 448 positions


def read_manifest(spec):
    path, _, n = spec.partition(":")
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    if n.startswith("~"):  # a random subset
        rows, n = random.sample(rows, min(int(n[1:]), len(rows))), ""
    for r in rows:
        r["audio_path"] = str((path.parent / r["audio"]).resolve())
        r["text"] = clean_label(r["text"], r["language"])  # house style, whatever the source
    return rows, int(n) if n else None


NOISE_FILES = []  # real background recordings (MUSAN noise / instrumental music), filled from --noise-dir


def real_noise(n, rng):
    """n samples of a random stretch of a random background recording, at 16 kHz."""
    import soundfile as sf
    import soxr

    path = NOISE_FILES[rng.integers(len(NOISE_FILES))]
    info = sf.info(path)
    need = int(n * info.samplerate / 16000) + 1
    x, sr = sf.read(path, start=int(rng.integers(0, max(1, info.frames - need))), frames=need, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = soxr.resample(x, sr, 16000)
    if len(x) < n:
        x = np.tile(x, n // max(len(x), 1) + 1)
    return x[:n].astype(np.float32)


def augment(audio, rng):
    """Make a clean recording sound more like real use: speed, loudness (sometimes far from the mic),
    room echo, phone line, background noise. Returns the audio and the speed factor (timestamps scale by 1/speed)."""
    import soxr

    speed = rng.uniform(0.9, 1.1) if rng.random() < 0.5 else 1.0
    if speed != 1.0:
        audio = soxr.resample(audio, 16000 * speed, 16000).astype(np.float32)
    # loudness: usually ordinary, one time in five very quiet, like someone far from the phone
    audio = audio * 10 ** ((rng.uniform(-30, -12) if rng.random() < 0.2 else rng.uniform(-8, 6)) / 20)
    if rng.random() < 0.3:  # small room: exponentially decaying noise as an impulse response
        n = int(16000 * rng.uniform(0.1, 0.4))
        ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.linspace(0, rng.uniform(4, 8), n))
        ir[0] = 1.0
        audio = np.convolve(audio, ir / np.abs(ir).sum() * 2)[: len(audio)].astype(np.float32)
    if rng.random() < 0.15:  # phone line: nothing above 4 kHz, a little clipping
        audio = soxr.resample(soxr.resample(audio, 16000, 8000), 8000, 16000).astype(np.float32)[: len(audio)]
        audio = np.clip(audio * 1.5, -0.5, 0.5)
    if rng.random() < 0.6:  # background: a real recording (MUSAN) or white/brown noise, 0-30 dB below the speech
        if NOISE_FILES and rng.random() < 0.7:
            noise = real_noise(len(audio), rng)
        else:
            noise = rng.standard_normal(len(audio)).astype(np.float32)
            if rng.random() < 0.5:
                noise = np.cumsum(noise); noise -= np.convolve(noise, np.ones(400) / 400, "same")
        snr = rng.uniform(0, 30)
        p_sig, p_noise = np.mean(audio ** 2) + 1e-9, np.mean(noise ** 2) + 1e-9
        audio = audio + noise * np.sqrt(p_sig / p_noise / 10 ** (snr / 10))
    return np.clip(audio, -1, 1).astype(np.float32), speed


class Clips(torch.utils.data.Dataset):
    def __init__(self, rows, processor, timestamp_frac=0.0, augment=0.0):
        self.rows, self.fe, self.tok = rows, processor.feature_extractor, processor.tokenizer
        self.timestamp_frac, self.augment = timestamp_frac, augment
        self.ts_begin = self.tok.convert_tokens_to_ids("<|notimestamps|>") + 1  # id of <|0.00|>, then one per 0.02 s
        assert self.tok.convert_tokens_to_ids("<|0.00|>") == self.ts_begin

    def __len__(self):
        return len(self.rows)

    def labels(self, r, speed=1.0):
        # Decoder target: <|lang|><|transcribe|><|notimestamps|> text <|endoftext|>
        # (<|startoftranscript|> is prepended by the model when it shifts labels right).
        # Some clips with known speech bounds are given in timestamp form instead,
        #   <|lang|><|transcribe|><|0.36|> text<|14.20|><|endoftext|>
        # so the model stays usable in players that decode with timestamps (whisper.cpp, WhisperKit).
        lang = r["language"]
        ids = self.tok.encode(" " + r["text"], add_special_tokens=False)[: MAX_LABEL - 5]
        task = self.tok.convert_tokens_to_ids([f"<|{lang}|>", "<|transcribe|>"])
        if self.timestamp_frac and "speech_end" in r and random.random() < self.timestamp_frac:
            ts = lambda t: self.ts_begin + min(1500, max(0, round(t / 0.02)))
            return task + [ts(r["speech_start"] / speed)] + ids + [ts(r["speech_end"] / speed)] + [self.tok.eos_token_id]
        return task + self.tok.convert_tokens_to_ids(["<|notimestamps|>"]) + ids + [self.tok.eos_token_id]

    def __getitem__(self, i):
        r = self.rows[i]
        audio, speed = load_audio(r["audio_path"]), 1.0
        if self.augment and random.random() < self.augment:
            audio, speed = augment(audio, np.random.default_rng(random.getrandbits(32)))
        feats = self.fe(audio[: 30 * 16000], sampling_rate=16000, return_tensors="np").input_features[0]
        return feats, self.labels(r, speed)


class CachedClips(Clips):
    """Same targets, but serves the frozen encoder's output from disk instead of audio features."""

    def __init__(self, rows, processor, cache, timestamp_frac=0.0):
        super().__init__(rows, processor, timestamp_frac)
        self.cache = cache

    def __getitem__(self, i):
        r = self.rows[i]
        return np.load(self.cache[r["audio_path"]]), self.labels(r)


def cache_encoder(model, encoder, processor, rows, cache_dir, device, dtype, batch=8, min_free_gb=1.5):
    """Run the frozen encoder once per unique clip and keep the result on disk (float16).

    With a frozen encoder this is the expensive part of every step, and the proofread clips
    are repeated several times per epoch, so computing it once makes training ~3x faster.
    """
    import hashlib

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({r["audio_path"] for r in rows})
    cache = {p: cache_dir / (hashlib.sha1(p.encode()).hexdigest() + ".npy") for p in paths}
    todo = [p for p in paths if not cache[p].exists()]
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(todo), batch):
            part = todo[i : i + batch]
            audio = [load_audio(p)[: 30 * 16000] for p in part]
            feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
            enc = encoder(feats.to(device, dtype)).last_hidden_state.to(torch.float16).cpu().numpy()
            if shutil.disk_usage(cache_dir).free / 1e9 < min_free_gb:
                raise SystemExit(f"stopping: disk nearly full while caching ({shutil.disk_usage(cache_dir).free / 1e9:.1f} GB free); "
                                 "rerun the same command to resume")
            for p, e in zip(part, enc):
                np.save(cache[p], e)
            done = i + len(part)
            if done % 80 < batch or done == len(todo):
                el = time.time() - t0
                print(f"  encoder cache {done}/{len(todo)}  eta {(len(todo) - done) * el / done / 60:.0f} min", flush=True)
    model.train()
    return cache


def collate(batch):
    feats = torch.from_numpy(np.stack([b[0] for b in batch]))  # audio features, or cached encoder output
    width = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (_, lab) in enumerate(batch):
        labels[i, : len(lab)] = torch.tensor(lab)
    return feats, labels


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32


@torch.no_grad()
def evaluate(model, processor, rows, device, dtype, batch=8):
    model.eval()
    hyps = []
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        audio = [load_audio(r["audio_path"])[: 30 * 16000] for r in part]
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        out = model.generate(input_features=feats.to(device, dtype), language=part[0]["language"],
                             task="transcribe", max_new_tokens=400, num_beams=1)
        hyps += processor.tokenizer.batch_decode(out, skip_special_tokens=True)
    model.train()
    lang = rows[0]["language"]
    wer, cer = word_char_errors([r["text"] for r in rows], [h.strip() for h in hyps], lang)
    return wer, cer, hyps


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--decoder-only", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--min-free-gb", type=float, default=1.5, help="stop caching cleanly if the disk gets this full")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="don't cache encoder outputs (decoder-only runs)")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=float, default=0.0,
                    help="share of clips given noise/echo/speed changes (needs the encoder in training, no cache)")
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute activations to fit encoder training in less memory")
    ap.add_argument("--workers", type=int, default=0, help="data-loading processes (CUDA boxes: 4-8)")
    ap.add_argument("--full", action="store_true",
                    help="train every weight instead of LoRA adapters (CUDA: fp32 master weights, bf16 compute, 8-bit AdamW)")
    ap.add_argument("--spec-augment", action="store_true", help="mask random time steps and mel bands while training")
    ap.add_argument("--noise-dir", nargs="*", default=[], help="folders of background recordings mixed into --augment clips")
    ap.add_argument("--save-every", type=int, default=0, help="full training: also save a checkpoint every N steps")
    ap.add_argument("--timestamp-frac", type=float, default=0.0,
                    help="share of clips with speech bounds trained in timestamp form (keeps timestamp decoding working)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    for d in args.noise_dir:
        NOISE_FILES.extend(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if args.noise_dir:
        print(f"background recordings for augmentation: {len(NOISE_FILES)}", flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = pick_device()
    print(f"device {device}, weights {dtype}", flush=True)

    processor = WhisperProcessor.from_pretrained(args.base)
    # Full training keeps fp32 master weights and computes in bf16 (autocast); LoRA runs in bf16 throughout.
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.full and device.type == "cuda")
    if args.full:
        dtype = torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=dtype).to(device)
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    if args.spec_augment:  # SpecAugment, as in the Whisper fine-tuning recipes: only active in train mode
        model.config.update(dict(apply_spec_augment=True, mask_time_prob=0.05, mask_time_length=10, mask_time_min_masks=2,
                                 mask_feature_prob=0.05, mask_feature_length=10, mask_feature_min_masks=0))

    if args.full:
        whisper = model
        if args.decoder_only:
            for p in model.model.encoder.parameters():
                p.requires_grad_(False)
        print(f"training all weights: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.0f}M", flush=True)
    else:
        layers = r"decoder\.layers\.\d+\." if args.decoder_only else r"(encoder|decoder)\.layers\.\d+\."
        target = rf".*{layers}((self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
        lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=target, bias="none")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        whisper = model.base_model.model
    if args.grad_ckpt:
        whisper.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        whisper.enable_input_require_grads()
    encoder = whisper.model.encoder

    train_rows = []
    for spec in args.train:
        rows, rep = read_manifest(spec)
        train_rows += rows * (rep or 1)
        print(f"train {spec}: {len(rows)} clips x{rep or 1}", flush=True)
    evals = []
    for spec in args.eval:
        rows, n = read_manifest(spec)
        evals.append((spec.partition(":")[0], rows[:n] if n else rows))

    report = {"base": args.base, "args": vars(args), "eval": {}}
    if not args.skip_baseline:
        for name, rows in evals:
            wer, cer, _ = evaluate(model, processor, rows, device, dtype)
            report["eval"][name] = {"before": {"wer": wer, "cer": cer}}
            print(f"before  {name}: WER {wer:.1f}%  CER {cer:.1f}%", flush=True)

    use_cache = args.decoder_only and not args.no_cache and not args.augment
    if use_cache:
        print("caching encoder outputs ...", flush=True)
        cache = cache_encoder(model, encoder, processor, train_rows, out / "enc_cache", device, dtype,
                              min_free_gb=args.min_free_gb)
        dataset = CachedClips(train_rows, processor, cache, args.timestamp_frac)
    else:
        dataset = Clips(train_rows, processor, args.timestamp_frac, args.augment)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                         collate_fn=collate, num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0)
    steps_per_epoch = len(loader) // args.accum
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = None
    if args.full and device.type == "cuda":
        try:  # 8-bit optimizer states: 2 bytes per weight instead of 8, so the whole model fits in 16 GB
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
        except ImportError:
            print("bitsandbytes not installed; using regular AdamW", flush=True)
    opt = opt or torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    print(f"{len(train_rows)} training clips, {total} steps of {args.batch * args.accum}", flush=True)

    model.train()
    step, t0, running = 0, time.time(), []
    stopped = None
    while step < total and not stopped:
        for i, (feats, labels) in enumerate(loader):
            feats, labels = feats.to(device, dtype), labels.to(device)
            with autocast:
                if use_cache:
                    loss = model(encoder_outputs=(feats,), labels=labels).loss
                elif args.decoder_only:
                    with torch.no_grad():  # frozen encoder: no activations kept for backward
                        enc = encoder(feats).last_hidden_state
                    loss = model(encoder_outputs=(enc,), labels=labels).loss
                else:
                    loss = model(input_features=feats, labels=labels).loss
            (loss / args.accum).backward()
            running.append(loss.item())
            if (i + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if args.full and args.save_every and step % args.save_every == 0 and step < total:
                ck = out / f"ckpt-{step}"
                model.save_pretrained(ck, state_dict={k: v.half() for k, v in model.state_dict().items()})
                processor.save_pretrained(ck)
                print(f"saved checkpoint {ck}", flush=True)
            if step % 10 == 0 or step == total:
                el = time.time() - t0
                print(f"step {step}/{total}  loss {np.mean(running):.3f}  {el / step:.1f}s/step  "
                      f"eta {(total - step) * el / step / 60:.0f} min  disk {free_gb(out):.1f} GB free", flush=True)
                running = []
            if step >= total:
                break

    if args.full:  # fp16 like the published checkpoints; the fp32 weights in memory stay as they are
        model.save_pretrained(out / "model", state_dict={k: v.half() for k, v in model.state_dict().items()})
        processor.save_pretrained(out / "model")
    else:
        model.save_pretrained(out / "adapter")
    if use_cache and not args.keep_cache and not stopped:
        shutil.rmtree(out / "enc_cache", ignore_errors=True)
    print(f"saved {'model' if args.full else 'adapter'} -> {out / ('model' if args.full else 'adapter')}"
          + (f"  (stopped early: {stopped})" if stopped else ""), flush=True)
    for name, rows in evals:
        with autocast:
            wer, cer, hyps = evaluate(model, processor, rows, device, dtype)
        report["eval"].setdefault(name, {})["after"] = {"wer": wer, "cer": cer}
        print(f"after   {name}: WER {wer:.1f}%  CER {cer:.1f}%", flush=True)
        with open(out / f"eval_{Path(name).parent.name}_{Path(name).stem}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h.strip()}, ensure_ascii=False) + "\n")
    report["steps"] = step
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

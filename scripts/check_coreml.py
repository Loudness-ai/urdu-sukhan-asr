"""On a Mac: compile a Core ML encoder made by scripts/export_coreml.py for whisper.cpp, and check it.

  python scripts/check_coreml.py models/coreml/coreml-encoder-student-e8d4.mlpackage --name student-e8d4

Compiles to models/ggml/ggml-<name>-encoder.mlmodelc (the name whisper.cpp looks for next to
ggml-<name>-q8_0.bin), runs it on the reference log-mel saved by the export (Neural Engine + GPU + CPU,
and CPU only), and compares with the PyTorch encoder output. Needs coremltools and Xcode's coremlc.
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlpackage")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "ggml"))
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    import coremltools as ct

    pkg = Path(args.mlpackage)
    final = Path(args.out_dir) / f"ggml-{args.name}-encoder.mlmodelc"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlc", "compile", str(pkg), tmp], check=True, stdout=subprocess.DEVNULL)
        built = next(Path(tmp).glob("*.mlmodelc"))
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(built), final)
    size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file())
    print(f"compiled: {final} ({size / 1e6:.0f} MB)")

    ref_dir = pkg.parent
    stem = pkg.stem.removeprefix("coreml-encoder-")
    mel = np.load(ref_dir / f"{stem}-ref-mel.npy")
    ref = np.load(ref_dir / f"{stem}-ref-out.npy")
    for units in ("ALL", "CPU_AND_NE", "CPU_ONLY"):
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(str(final), compute_units=getattr(ct.ComputeUnit, units))
        load = time.perf_counter() - t0
        out = model.predict({"logmel_data": mel})["output"]  # first call also warms up
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            out = model.predict({"logmel_data": mel})["output"]
            times.append(time.perf_counter() - t0)
        out = np.asarray(out, dtype=np.float32).reshape(ref.shape)
        cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref)))
        rel = float(np.abs(out - ref).mean() / np.abs(ref).mean())
        print(f"{units:10s} load {load:5.1f} s  encode {min(times) * 1000:6.0f} ms per 30 s  "
              f"vs PyTorch: cosine {cos:.5f}, mean rel. error {rel:.4f}")


if __name__ == "__main__":
    main()

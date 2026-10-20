#!/usr/bin/env bash
# One-time setup of a rented Linux GPU box (Ubuntu with an NVIDIA driver) for training.
# Run on the server from the project folder after copying the code there:
#   bash scripts/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi -L || { echo "no NVIDIA GPU/driver found"; exit 1; }
# ffmpeg is optional (the clips are FLAC/WAV, which soundfile reads with its bundled libsndfile).
if command -v apt-get >/dev/null && { [ "$(id -u)" -eq 0 ] || sudo -n true 2>/dev/null; }; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq ffmpeg >/dev/null || true
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d .venv ] || uv venv --python 3.12 .venv
# PyPI's torch wheel targets the newest CUDA; rented boxes often have an older driver, so take the
# build that matches it (nvidia-smi's "CUDA Version") from PyTorch's own index first.
cuda=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$" || true)
case "$cuda" in 12.6*|12.7*) idx=cu126 ;; 12.8*|12.9*) idx=cu128 ;; 13.*) idx=cu130 ;; *) idx=cu126 ;; esac
torch_pin=$(grep '^torch==' requirements.txt)
uv pip install --python .venv/bin/python -q -r requirements.txt
if ! .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  uv pip install --python .venv/bin/python -q --reinstall-package torch "$torch_pin" --index-url "https://download.pytorch.org/whl/$idx"
fi
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

if [ ! -f models/hf/whisper-large-v3-turbo/model.safetensors ]; then
  .venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-large-v3-turbo", local_dir="models/hf/whisper-large-v3-turbo")
EOF
fi
echo "ready"

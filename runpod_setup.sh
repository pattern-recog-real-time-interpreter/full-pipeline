#!/usr/bin/env bash
# One-shot RunPod setup for the Thai → English voice translation pipeline.
#
# On a fresh RunPod PyTorch 2.x pod, paste this in the web terminal:
#
#   bash <(curl -sL https://raw.githubusercontent.com/pattern-recog-real-time-interpreter/full-pipeline/main/runpod_setup.sh)
#
# Or, if the repo is already cloned:
#
#   cd full-pipeline && bash runpod_setup.sh

set -euo pipefail

REPO_URL="https://github.com/pattern-recog-real-time-interpreter/full-pipeline.git"
REPO_DIR="full-pipeline"

echo "==> [1/6] System packages (ffmpeg, portaudio)"
apt-get update -qq
apt-get install -y -qq libportaudio2 ffmpeg

echo "==> [2/6] Install uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> [3/6] Repo"
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git pull
elif [ -d ".git" ] && [ -f "app.py" ]; then
    : # already inside the repo
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Install into the RunPod image's system Python so the pre-installed CUDA torch
# is reused (uv sync would create a fresh venv with CPU-only torch).
echo "==> [4/6] Python dependencies (~10 min, nemo_toolkit is slow)"
uv pip install --system -r requirements.txt

echo "==> [5/6] Downloading models (~2-3 GB)"
python setup.py

echo "==> [6/6] Launching Gradio on port 7860 (GPU)"
echo
echo "    Open the RunPod 'HTTP Service' for port 7860 to access the UI."
echo "    Press Ctrl+C to stop."
echo
python app.py --device cuda

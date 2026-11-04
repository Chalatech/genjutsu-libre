#!/usr/bin/env bash
# Idempotent, user-space setup of the GPU host workers. Run it ON the GPU host,
# from a copy of this folder. It only writes inside $GPU_DIR. No sudo, no apt.
#
#   rsync -az gpu_workers/ "$GPU_HOST:$GPU_DIR/workers/"
#   ssh "$GPU_HOST" "GPU_DIR=$GPU_DIR bash $GPU_DIR/workers/setup_gpu.sh"
#
# Optional environment (also read from $GPU_DIR/.env, keep it chmod 600):
#   SEGMENT_MODEL  sam3 (default) | sam2 (fallback, no gated weights needed)
#   SAM3_WEIGHTS   folder with Sam3VideoModel weights in transformers format
#                  (default $GPU_DIR/models/sam3)
#   SAM3_SOURCE    existing local copy of those weights, copied once into SAM3_WEIGHTS
#   HF_TOKEN       your own Hugging Face token, only to download facebook/sam3
#   TORCH_INDEX    PyTorch wheel index (default: CUDA 13.0 builds)
#
# If no CUDA build of torch works on the host's GPU, the documented fallback is
# NVIDIA's PyTorch container (nvcr.io/nvidia/pytorch, recent tag) run with
# `docker run --rm --gpus all`, only when the user can already use docker
# without sudo; install the packages of requirements-gpu.txt inside it.
set -euo pipefail

: "${GPU_DIR:?Set GPU_DIR to the working folder on this host}"
GPU_DIR="${GPU_DIR/#\~/$HOME}"
if [ -f "$GPU_DIR/.env" ]; then set -a; . "$GPU_DIR/.env"; set +a; fi
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
SEGMENT_MODEL="${SEGMENT_MODEL:-sam3}"
SAM3_WEIGHTS="${SAM3_WEIGHTS:-$GPU_DIR/models/sam3}"
SAM3_WEIGHTS="${SAM3_WEIGHTS/#\~/$HOME}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Pinned upstream commits (checked 2026-11-04).
VDA_REPO=https://github.com/DepthAnything/Video-Depth-Anything.git
VDA_COMMIT=4f5ae23172ba60fd7bc11ef671cca678842c7072
SAM2_REPO=https://github.com/facebookresearch/sam2.git
SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4

mkdir -p "$GPU_DIR"/{bin,models/torch,hf-cache,src,workers,tmp}
export HF_HOME="$GPU_DIR/hf-cache"
export TORCH_HOME="$GPU_DIR/models/torch"  # htdemucs weights
export UV_CACHE_DIR="$GPU_DIR/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$GPU_DIR/.uv-python"
export TMPDIR="$GPU_DIR/tmp"

# 1. uv, installed inside GPU_DIR (not in ~/.local/bin).
if [ ! -x "$GPU_DIR/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$GPU_DIR/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
UV="$GPU_DIR/bin/uv"

# 2. venv with Python 3.12 (the system one when present, else a uv-managed one).
if [ ! -x "$GPU_DIR/.venv/bin/python" ]; then
  if [ -x /usr/bin/python3.12 ]; then "$UV" venv --python /usr/bin/python3.12 "$GPU_DIR/.venv"
  else "$UV" venv --python 3.12 "$GPU_DIR/.venv"; fi
fi
PY="$GPU_DIR/.venv/bin/python"
PIP=("$UV" pip install --python "$PY")

# 3. torch with CUDA for this architecture (aarch64/sbsa builds live on the same index).
cuda_ok() { "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }
if ! cuda_ok; then
  "${PIP[@]}" --index-url "$TORCH_INDEX" torch torchvision
  cuda_ok || { echo "torch from $TORCH_INDEX has no working CUDA here; see the container fallback above." >&2; exit 1; }
fi
# Never let a later install swap the CUDA build of torch for another one.
"$PY" - > "$TMPDIR/torch-pins.txt" <<'PY'
import importlib.metadata as md
for name in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY

# 4. Python dependencies (transformers ships Sam3VideoModel).
"${PIP[@]}" --constraint "$TMPDIR/torch-pins.txt" -r "$HERE/requirements-gpu.txt"

# 5. Pinned source checkouts (Video-Depth-Anything is used from source, not installed).
fetch() { # name repo commit
  local dir="$GPU_DIR/src/$1"
  if [ ! -d "$dir/.git" ]; then git clone --quiet "$2" "$dir"; fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$3" ]; then
    git -C "$dir" fetch --quiet origin "$3" 2>/dev/null || true
    git -C "$dir" checkout --quiet "$3"
  fi
}
fetch video-depth-anything "$VDA_REPO" "$VDA_COMMIT"

# 6. Weights. Video-Depth-Anything Small is Apache-2.0; Large is CC-BY-NC and is NOT fetched.
mkdir -p "$GPU_DIR/models/video-depth-anything"
VDA_W="$GPU_DIR/models/video-depth-anything/video_depth_anything_vits.pth"
if [ ! -s "$VDA_W" ]; then
  curl -L --fail -s -o "$VDA_W.part" \
    https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
  mv "$VDA_W.part" "$VDA_W"
fi

if [ "$SEGMENT_MODEL" = "sam3" ]; then
  if [ ! -s "$SAM3_WEIGHTS/model.safetensors" ]; then
    if [ -n "${SAM3_SOURCE:-}" ]; then
      SAM3_SOURCE="${SAM3_SOURCE/#\~/$HOME}"
      mkdir -p "$SAM3_WEIGHTS"
      cp -r "$SAM3_SOURCE/." "$SAM3_WEIGHTS/"
    elif [ -n "${HF_TOKEN:-}" ]; then
      HF_TOKEN="$HF_TOKEN" "$PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3', local_dir=sys.argv[1], allow_patterns=['*.json', '*.txt', 'model.safetensors'])" "$SAM3_WEIGHTS"
    else
      echo "SEGMENT_MODEL=sam3 needs SAM3_SOURCE, or HF_TOKEN (your own account, access granted on facebook/sam3). Or set SEGMENT_MODEL=sam2." >&2
    fi
  fi
else
  fetch sam2 "$SAM2_REPO" "$SAM2_COMMIT"
  # SAM2_BUILD_CUDA=0: skip the optional CUDA extension (no nvcc needed).
  "${PIP[@]}" setuptools wheel
  SAM2_BUILD_CUDA=0 "${PIP[@]}" --constraint "$TMPDIR/torch-pins.txt" --no-build-isolation -e "$GPU_DIR/src/sam2"
  mkdir -p "$GPU_DIR/models/sam2"
  SAM2_W="$GPU_DIR/models/sam2/sam2.1_hiera_small.pt"
  if [ ! -s "$SAM2_W" ]; then
    curl -L --fail -s -o "$SAM2_W.part" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
    mv "$SAM2_W.part" "$SAM2_W"
  fi
fi

# htdemucs weights go to $TORCH_HOME (CPU only, no GPU used here).
"$PY" -c 'from demucs.pretrained import get_model; get_model("htdemucs")' >/dev/null

# 7. Sync the workers next to the venv and check the GPU.
if [ "$HERE" != "$GPU_DIR/workers" ]; then
  rsync -a --exclude __pycache__ "$HERE/" "$GPU_DIR/workers/"
fi
SEGMENT_MODEL="$SEGMENT_MODEL" SAM3_WEIGHTS="$SAM3_WEIGHTS" GPU_DIR="$GPU_DIR" \
  nice -n 10 "$PY" "$GPU_DIR/workers/check.py"
echo "disk used: $(du -sh "$GPU_DIR" | cut -f1)"

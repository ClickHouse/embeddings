#!/bin/bash
# Self-bootstrapping setup in a venv (avoids fighting rpm-managed system packages).
# Picks Python >=3.10 because Nemotron-VL's remote code uses 3.10+ 'X | Y' type unions
# (the other 3 models work on 3.9). --system-site-packages reuses a DLAMI's torch if present.
# Requires an NVIDIA driver on the host.
set -e

# --- 1. choose a Python >= 3.10 -------------------------------------------------
if python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
  PY=python3
else
  echo "system python3 is $(python3 -V 2>&1) (<3.10); Nemotron-VL needs >=3.10 — locating/installing a newer Python…"
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10; do command -v "$c" >/dev/null 2>&1 && { PY=$c; break; }; done
  if [ -z "$PY" ]; then
    command -v dnf     >/dev/null 2>&1 && dnf install -y python3.11 >/dev/null 2>&1 || true
    command -v apt-get >/dev/null 2>&1 && { apt-get update -y >/dev/null 2>&1; apt-get install -y python3.11 python3.11-venv >/dev/null 2>&1; } || true
    for c in python3.12 python3.11 python3.10; do command -v "$c" >/dev/null 2>&1 && { PY=$c; break; }; done
  fi
  if [ -z "$PY" ]; then
    PY=python3
    echo "WARNING: could not obtain Python>=3.10. Continuing with $(python3 -V 2>&1):"
    echo "         the 3 non-Nemotron models work -> run  './run.sh <ngpu> siglip2,clip,nomic'"
    echo "         Nemotron-VL will fail until you install python3.11 (e.g. dnf install -y python3.11) and re-run setup."
  fi
fi
echo "using $($PY -V 2>&1) at $(command -v "$PY")"

# --- 2. venv (recreate if a stale <3.10 one exists) -----------------------------
VENV="$PWD/venv"
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)' 2>/dev/null; then
  echo "existing venv is <3.10 — recreating with $PY"; rm -rf "$VENV"
fi
$PY -m pip --version >/dev/null 2>&1 || $PY -m ensurepip --upgrade || true
[ -x "$VENV/bin/python" ] || $PY -m venv --system-site-packages "$VENV" || {
  echo "ERROR: venv creation failed — install the venv module (AL2023: dnf install -y python3.11 ; Ubuntu: apt-get install -y python3.11-venv)"; exit 1; }
PYV="$VENV/bin/python"
$PYV -m pip install --upgrade pip
PI(){ $PYV -m pip install -U "$@"; }
echo "venv python: $($PYV -V 2>&1)"

command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: nvidia-smi not found — no NVIDIA driver. Use a GPU AMI."

# --- 3. torch (only if not already visible) + deps ------------------------------
if $PYV -c "import torch" 2>/dev/null; then
  echo "torch already visible: $($PYV -c 'import torch;print(torch.__version__)')"
else
  echo "installing torch/torchvision (CUDA 12.4 wheels)…"
  PI torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
# NOTE: pin transformers <5 — the nomic-vision & Nemotron-VL trust_remote_code models were written
# for the 4.x API; transformers 5.x renamed internals (e.g. _tied_weights_keys) and fails to load them.
PI "transformers>=4.49,<5" open_clip_torch pillow pyarrow boto3 requests einops sentencepiece protobuf "numpy<2"

# --- 4. clickhouse client (OPTIONAL) -------------------------------------------
command -v curl >/dev/null 2>&1 && (curl -s https://clickhouse.com/ | sh || true)

# --- 5. verify (non-fatal) -----------------------------------------------------
$PYV -c "import torch, transformers, open_clip, pyarrow, boto3, requests; \
print('python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.device_count(), 'gpus')" || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo "setup done — run.sh auto-uses $VENV/bin/python."

#!/bin/bash
# Install GPU-UMAP deps in a SEPARATE venv (venv_umap) so this never disturbs the embedding job's venv
# (installing RAPIDS could upgrade numpy/etc. that the running siglip2 workers import). Safe to run anytime.
set -e
cd "$(dirname "$0")/.."
# RAPIDS needs Python >=3.10 (this box has 3.9 and 3.13; use 3.13 — cuml-cu12 26.x ships cp313 wheels).
PYBASE=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10)
[ -n "$PYBASE" ] || { echo "ERROR: need python>=3.10 for RAPIDS (found only $(python3 -V 2>&1))"; exit 1; }
echo "base python: $($PYBASE -V)"
[ -x venv_umap/bin/python ] || "$PYBASE" -m venv venv_umap
V=venv_umap/bin/python
"$V" -m pip install -U pip wheel
# RAPIDS cuML + cuPy (CUDA 12 wheels from NVIDIA's index), ClickHouse client, arrow. Let deps pick numpy.
# cuML UMAP/PCA run on the GPU; clickhouse-connect streams embeddings in/out.
"$V" -m pip install --extra-index-url=https://pypi.nvidia.com \
    cuml-cu12 cupy-cuda12x "clickhouse-connect>=0.8" pyarrow
echo "=== verify ==="
"$V" - <<'PY'
import cuml, cupy, clickhouse_connect, numpy
print("cuml", cuml.__version__, "| cupy", cupy.__version__, "| numpy", numpy.__version__)
from cuml.manifold import UMAP; from cuml.decomposition import PCA
print("cuML UMAP + PCA import OK")
PY
echo "setup_umap done -> venv_umap/bin/python"

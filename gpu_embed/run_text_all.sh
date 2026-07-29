#!/bin/bash
# Drive many local text-embedding models smallest-first, one at a time (each uses all GPUs).
# Each model: embed HN text on all GPUs -> stage -> build hackernews_embeddings_<model> (QBit).
# A model that fails to load is skipped; the rest continue. Fully resumable.
# Usage: CH_HOST=.. CH_PW=.. ./run_text_all.sh [model1 model2 ...]
set -u
: "${CH_HOST:?}"; : "${CH_PW:?}"
export CH_HOST CH_PW CH_USER

# smallest -> largest (see text_models.py)
DEFAULT_MODELS="minilm arctic_xs bge_small gte_small e5_small granite_small \
jina_small arctic_m bge_base gte_base e5_base nomic embeddinggemma \
kalm bge_large e5_large jina_v3 nemotron"
MODELS=${*:-$DEFAULT_MODELS}
PORT=${CH_PORT:-8443}
ch() { curl -sS "https://${CH_HOST}:${PORT}/" --user "${CH_USER:-default}:${CH_PW}" --data-binary "$1"; }

echo "########## driver start: $MODELS ##########"
for M in $MODELS; do
  DST=hackernews_embeddings_${M}
  N=$(ch "SELECT count() FROM ${DST}" 2>/dev/null || echo 0)
  if [[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -gt 1000000 ]; then
    echo "########## $M already built ($N rows) — skip ##########"
    continue
  fi
  echo "########## $M : starting $(date -u +%H:%M:%S) ##########"
  ./run_text.sh "$M" || echo "########## $M : run_text.sh returned non-zero ##########"
  echo "########## $M : finished $(date -u +%H:%M:%S) ##########"
done
echo "########## driver done ##########"

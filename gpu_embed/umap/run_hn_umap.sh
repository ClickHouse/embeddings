#!/bin/bash
# Build UMAP atlas for HackerNews text-embedding variants -> ClickHouse
# (hackernews_umap_<model>_{2d,3d}(id, x,y[,z], color), keyed by id).
#
# PHASE 1: PCA-reduce each model to 64d (cached memmap in umap_work/), max 1 job/GPU.
# PHASE 2: fit+transform the (model x {2d,3d}) UMAP jobs, max 1 job/GPU.
# Resumable: reduce cache persists; a re-run re-does only the (fast) UMAP transform.
#
# Usage: CH_HOST=.. CH_PW=.. ./umap/run_hn_umap.sh ["model1 model2 ..."] [extra args to hn_umap_atlas.py]
# Default models = all 17 built HN embedding tables (nemotron excluded — not built).
set -e
cd "$(dirname "$0")/.."
: "${CH_HOST:?export CH_HOST=...}"; : "${CH_PW:?export CH_PW=...}"; export CH_HOST CH_PW CH_USER
V="$PWD/venv_umap/bin/python"; [ -x "$V" ] || { echo "run umap/setup_umap.sh first"; exit 1; }
read -r -a MODELS <<< "${1:-minilm arctic_xs bge_small gte_small e5_small granite_small jina_small arctic_m bge_base gte_base e5_base nomic embeddinggemma kalm bge_large e5_large jina_v3}"
EXTRA=("${@:2}")
NGPU=$(nvidia-smi -L | wc -l)

declare -a GPU_PID
launch_on_free(){   # launch "$@" (args to hn_umap_atlas.py) on the next free GPU, max 1 job/GPU
  local g pid
  while :; do
    for g in $(seq 0 $((NGPU - 1))); do
      pid=${GPU_PID[$g]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        local log="umap/hnumap_${1}_${LBL}.log"
        CUDA_VISIBLE_DEVICES=$g nohup "$V" umap/hn_umap_atlas.py --model "$@" "${EXTRA[@]}" > "$log" 2>&1 &
        GPU_PID[$g]=$!
        echo "  $1/$LBL -> gpu$g (pid ${GPU_PID[$g]}, log $log)"
        return
      fi
    done
    sleep 5
  done
}

# PHASE 1 (PCA reduce) is READ-bound on the remote ClickHouse (~33k rows/s/connection, ceiling ~150k/s),
# not GPU-bound. Run PCONC concurrent reduces (default 2*NGPU) to saturate read bandwidth; GPUs stay idle here.
PCONC=${PCONC:-$((2 * NGPU))}
declare -a RPID
reduce_when_slot(){   # launch a reduce; block until < PCONC reduces are running
  local m=$1 g running
  while :; do
    running=0; for p in "${RPID[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && running=$((running+1)); done
    [ "$running" -lt "$PCONC" ] && break; sleep 3
  done
  g=$(( ${#RPID[@]} % NGPU ))
  CUDA_VISIBLE_DEVICES=$g nohup "$V" umap/hn_umap_atlas.py --model "$m" --reduce-only "${EXTRA[@]}" \
      > "umap/hnumap_${m}_reduce.log" 2>&1 &
  RPID+=($!); echo "  reduce $m -> gpu$g (pid ${RPID[-1]})"
}

echo "==== $(date '+%F %T') PHASE 1: PCA reduce (${#MODELS[@]} models), $PCONC concurrent (read-bound) ===="
for m in "${MODELS[@]}"; do reduce_when_slot "$m"; done
wait
echo "==== $(date '+%F %T') reduce complete ===="

echo "==== $(date '+%F %T') PHASE 2: UMAP fit+transform ($(( ${#MODELS[@]} * 2 )) jobs), max $NGPU concurrent ===="
for m in "${MODELS[@]}"; do
  for d in 2d 3d; do LBL=$d; launch_on_free "$m" --dims "$d"; done
done
wait
echo "ALL HN UMAP DONE $(date '+%F %T')"

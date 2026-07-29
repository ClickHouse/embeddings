#!/bin/bash
# Accelerated WonderfulWeb IMAGE pipeline: WPG workers per physical GPU (read-bound, so more concurrent
# shard reads ~= proportional speedup). Resumable (per-shard .done skip) + supervised.
# Usage: WW_HOST/WW_PW + CH_HOST/CH_PW in env; ./run_web_img.sh [workers_per_gpu]
set -u
: "${WW_HOST:?}"; : "${WW_PW:?}"; : "${CH_HOST:?}"; : "${CH_PW:?}"
export WW_HOST WW_PW CH_HOST CH_PW CH_USER HF_HUB_DISABLE_PROGRESS_BARS=1
# cap torch/BLAS threads: N image workers each load 3 vision models; without this their idle intra-op
# threads oversubscribe the CPU (thrash) and starve everything. Reads (not CPU) are the bottleneck.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$(dirname "$0")"
PY="$PWD/venv/bin/python"
NGPU=$(nvidia-smi -L | wc -l)
WPG="${1:-3}"
NW=$((WPG * NGPU))
MAXTRIES=${MAXTRIES:-1000}

supervise(){
  set +e
  local w=$1
  local phys=$((w % NGPU))
  local log="web_img_w${w}.log" tries=0
  while :; do
    CUDA_VISIBLE_DEVICES=$phys "$PY" embed_web_images.py --gpu "$w" --ngpu "$NW" --batch 64 >>"$log" 2>&1
    rc=$?; [ $rc -eq 0 ] && { echo "[img w$w] finished ok" >>"$log"; break; }
    tries=$((tries+1)); [ $tries -ge $MAXTRIES ] && { echo "[img w$w] gave up after $tries" >>"$log"; break; }
    echo "[img w$w] exited rc=$rc, relaunch #$tries in 15s" >>"$log"; sleep 15
  done
}

pids=()
for w in $(seq 0 $((NW-1))); do
  : > "web_img_w${w}.log"; supervise "$w" & pids+=($!)
done
echo "launched $NW image workers ($WPG/GPU across $NGPU GPUs). monitor: tail -f web_img_w0.log"
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL IMAGE WORKERS DONE"

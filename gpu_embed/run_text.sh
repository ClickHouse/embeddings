#!/bin/bash
# Embed HN `text` with ONE local model across all GPUs, then build the QBit table.
# Resumable (per-chunk .done markers) and self-healing (relaunches a crashed worker).
# Usage: CH_HOST=.. CH_PW=.. ./run_text.sh <model> [ngpu] [batch]
set -e
: "${CH_HOST:?export CH_HOST=<clickhouse host>}"; : "${CH_PW:?export CH_PW=<password>}"
export CH_HOST CH_PW CH_USER
MODEL=${1:?usage: ./run_text.sh <model> [ngpu] [batch]}
NGPU=${2:-$(nvidia-smi -L | wc -l)}
BATCH=${3:-512}
PY="$([ -x "$PWD/venv/bin/python" ] && echo "$PWD/venv/bin/python" || echo python3)"
OUT=${OUT:-$PWD/emb_text}
mkdir -p "$OUT"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation for large models (nemotron 7B)
[ -f "$PWD/.hf_token" ] && export HF_TOKEN="$(cat "$PWD/.hf_token")"   # for gated models (embeddinggemma)

MAX_TRIES=${MAX_TRIES:-5}
supervise() {   # relaunch a worker on non-zero exit (OOM / spot / transient CH error)
  set +e
  local gpu=$1 log=$2 tries=0
  while :; do
    "$PY" embed_text.py --model "$MODEL" --gpu "$gpu" --ngpu "$NGPU" --batch "$BATCH" --out "$OUT" >>"$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { echo "[gpu$gpu] worker finished ok" >>"$log"; break; }
    tries=$((tries+1))
    if [ $tries -ge $MAX_TRIES ]; then
      echo "[gpu$gpu] worker failed rc=$rc after $tries tries; giving up" >>"$log"; break
    fi
    echo "[gpu$gpu] worker exited rc=$rc (relaunch #$tries in 10s)" >>"$log"
    sleep 10
  done
}

echo "=== model=$MODEL  ngpu=$NGPU  batch=$BATCH  out=$OUT  host=$CH_HOST ==="
pids=()
for g in $(seq 0 $((NGPU-1))); do
  log="$PWD/text_${MODEL}_gpu${g}.log"
  : > "$log"
  supervise "$g" "$log" &
  pids+=($!)
  echo "  gpu$g -> $log"
  sleep 2   # stagger so gpu0 creates the stage table before others check
done
echo "workers launched. monitor: tail -f text_${MODEL}_gpu0.log"
for p in "${pids[@]}"; do wait "$p"; done
echo "=== all workers done for $MODEL; finalizing ==="
./finalize_text.sh "$MODEL"

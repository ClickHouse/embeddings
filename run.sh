#!/bin/bash
# Fan out the 4096 hash-prefixes across GPU workers. Resumable (skips done prefixes).
# Each worker uploads every prefix to ClickHouse as soon as it's embedded (needs CH_HOST + CH_PW).
# Usage: CH_HOST=... CH_PW=... ./run.sh [num_gpus] [models] [batch]
# PROCS_PER_GPU=N runs N workers per GPU (default 1). Cheap models (nomic/clip) starve a single
# stream — the GPU idles waiting for JPEG decode — so PROCS_PER_GPU=2 ~doubles throughput there.
set -e
: "${CH_HOST:?export CH_HOST=<clickhouse host>}"; : "${CH_PW:?export CH_PW=<password>}"
export CH_HOST CH_PW CH_USER
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # reduce VRAM fragmentation
PY="$([ -x "$PWD/venv/bin/python" ] && echo "$PWD/venv/bin/python" || echo python3)"   # use setup.sh's venv
NGPU=${1:-$(nvidia-smi -L | wc -l)}
MODELS=${2:-siglip2,clip,nomic,nemotron}
BATCH=${3:-256}
OUT=${OUT:-$PWD/emb}          # only holds tiny per-prefix .done markers now (embeddings stream to CH)
PPG=${PROCS_PER_GPU:-1}
NW=$((NGPU * PPG))            # total workers; worker w pinned to GPU (w % NGPU)
mkdir -p "$OUT"
"$PY" -c "print('\n'.join('%03x'%i for i in range(4096)))" > prefixes.txt
FIRST_MODEL=${MODELS%%,*}
MAX_RELAUNCH=${MAX_RELAUNCH:-50}   # cap per-worker relaunches so a poison state can't loop forever

# Supervise one worker: (re)launch until it exits cleanly (rc=0 = its prefix list is exhausted) or all
# 4096 prefixes are done. A crash/OOM-kill (rc!=0) is auto-relaunched; the job is resumable via .done
# skips, so it just picks up where it left off. This is why a silently-killed worker no longer idles a GPU.
supervise(){
  local w=$1 gpu=$2 tries=0 rc ndone
  set +e                                           # CRITICAL: run.sh runs under `set -e`; without this a
                                                   # crashing worker's non-zero exit would abort the supervisor
                                                   # itself (before relaunch) and silently idle the GPU.
  : > "gpu$w.log"                                  # fresh log at pass start (relaunches append)
  while :; do
    rc=0
    CUDA_VISIBLE_DEVICES=$gpu "$PY" embed_images.py \
        --prefix-file "prefixes.$w.txt" --out "$OUT" --models "$MODELS" --batch "$BATCH" \
        >> "gpu$w.log" 2>&1 || rc=$?              # `|| rc=$?` keeps set -e from firing and captures the code
    ndone=$(ls "$OUT/$FIRST_MODEL"/*.done 2>/dev/null | wc -l)
    if [ "$rc" -eq 0 ] || [ "$ndone" -ge 4096 ]; then
      echo "[supervisor $(date '+%F %T')] worker$w done (rc=$rc, $FIRST_MODEL=$ndone/4096)" >> "gpu$w.log"; break
    fi
    tries=$((tries+1))
    if [ "$tries" -ge "$MAX_RELAUNCH" ]; then
      echo "[supervisor $(date '+%F %T')] worker$w gave up after $MAX_RELAUNCH relaunches (rc=$rc)" >> "gpu$w.log"; break
    fi
    echo "[supervisor $(date '+%F %T')] worker$w died rc=$rc ($FIRST_MODEL=$ndone/4096); relaunch #$tries in 15s" >> "gpu$w.log"
    sleep 15
  done
}

echo "launching $NW workers ($NGPU GPUs x $PPG procs), models=$MODELS, batch=$BATCH, out=$OUT (auto-relaunch on)"
# Dynamic load balancing: give EVERY worker the full 4096-prefix list, just rotated to a different start.
# Prefix sizes vary 5..27k images, so a static NR%NW split leaves fast GPUs idle for hours at the tail.
# With rotation + the per-prefix .done skip (a cheap stat, no S3), each worker keeps finding undone work
# and no GPU goes idle until ALL prefixes are done; overlap only at the very tail (dedup-safe, wasted compute only).
for w in $(seq 0 $((NW-1))); do
  gpu=$((w % NGPU))
  awk -v w="$w" -v nw="$NW" '{a[NR]=$0} END{n=NR; off=int(w*n/nw); for(i=0;i<n;i++) print a[((i+off)%n)+1]}' \
      prefixes.txt > "prefixes.$w.txt"
  echo "  worker$w -> gpu$gpu, start=$(head -1 prefixes.$w.txt) (log gpu$w.log)"
  supervise "$w" "$gpu" &                          # background supervisor (auto-relaunches its worker)
done
echo "running. monitor: tail -f gpu0.log ; progress: ls $OUT/siglip2 | wc -l  (out of 4096)"
wait
echo "ALL GPU WORKERS DONE"

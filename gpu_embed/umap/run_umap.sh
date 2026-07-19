#!/bin/bash
# Build the UMAP atlas for all 3 models -> ClickHouse (mmcommons.umap_<model>_{2d,3d}).
#
# DEFAULTS (confirmed with user):
#   * color   = one extra UMAP dimension -> maps to HUE at viz time (stored raw in `color`; see hue query below)
#   * scale   = fit UMAP on a 10M sample, TRANSFORM all ~99.6M (cuML transform = project rest by nearest neighbors)
#   * reduce  = PCA -> 64d first
#   * hardware= this box; PHASE 1 reduces the 3 models in parallel, PHASE 2 runs the 6 (model x dim)
#               UMAP jobs across ALL GPUs, one job per GPU (a spare GPU is used to shorten the long pole).
#   * output  = 6 tables umap_<model>_{2d,3d}(md5, x,y[,z], color), keyed by md5
#
# Hue at viz time (color is a raw, unbounded UMAP axis -> rank/quantile-normalize to [0,1) then x360):
#   SELECT md5, x, y, (color - q01)/(q99 - q01) AS hue01
#   FROM umap_nomic_2d, (SELECT quantile(0.01)(color) q01, quantile(0.99)(color) q99 FROM umap_nomic_2d)
#
# Prereq: bash umap/setup_umap.sh  (RAPIDS in venv_umap). Creds: CH_HOST + CH_PW in env.
set -e
cd "$(dirname "$0")/.."
: "${CH_HOST:?export CH_HOST=...}"; : "${CH_PW:?export CH_PW=...}"; export CH_HOST CH_PW CH_USER
V="$PWD/venv_umap/bin/python"; [ -x "$V" ] || { echo "run umap/setup_umap.sh first"; exit 1; }
read -r -a MODELS <<< "${1:-nomic clip siglip2}"
EXTRA=("${@:2}")
NGPU=$(nvidia-smi -L | wc -l)

echo "==== $(date '+%F %T') PHASE 1: PCA reduce (${MODELS[*]}), one per GPU ===="
i=0
for m in "${MODELS[@]}"; do
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup "$V" umap/umap_atlas.py --model "$m" --reduce-only "${EXTRA[@]}" \
      > "umap/reduce_$m.log" 2>&1 &
  echo "  reduce $m -> gpu$((i % NGPU)) (log umap/reduce_$m.log)"
  i=$((i + 1))
done
wait
echo "==== $(date '+%F %T') reduce complete ===="

# PHASE 2: fan the 6 (model x dim) UMAP jobs across all GPUs, max 1 job/GPU (avoids CUDA-context contention).
declare -a GPU_PID
launch_on_free(){
  local m=$1 d=$2 g pid
  while :; do
    for g in $(seq 0 $((NGPU - 1))); do
      pid=${GPU_PID[$g]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        CUDA_VISIBLE_DEVICES=$g nohup "$V" umap/umap_atlas.py --model "$m" --dims "$d" "${EXTRA[@]}" \
            > "umap/umap_${m}_${d}.log" 2>&1 &
        GPU_PID[$g]=$!
        echo "  $m/$d -> gpu$g (pid ${GPU_PID[$g]}, log umap/umap_${m}_${d}.log)"
        return
      fi
    done
    sleep 5
  done
}
echo "==== $(date '+%F %T') PHASE 2: UMAP fit+transform (6 jobs across $NGPU GPUs) ===="
for m in "${MODELS[@]}"; do for d in 2d 3d; do launch_on_free "$m" "$d"; done; done
wait
echo "ALL UMAP DONE $(date '+%F %T')"

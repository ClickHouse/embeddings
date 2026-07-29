#!/bin/bash
# Final UMAP batch: qwen3_8b (HN) + 7 WonderfulWeb variants. Two phases like run_hn_umap.sh:
#   PHASE 1 reduce (IO-bound, run concurrent, 1/GPU to bound PCA GPU mem)
#   PHASE 2 UMAP 2d+3d (GPU-bound, 1 job/GPU)
# Resumable: reduce cache (umap_work/*.meta) persists; UMAP TRUNCATEs+refills its table each run.
# Env: CH_HOST/CH_PW/CH_USER (embedding service, hvdvsqo23t). qwen3 also reads there.
set -u
: "${CH_HOST:?}"; : "${CH_PW:?}"
export CH_HOST CH_PW CH_USER
cd "$(dirname "$0")"
PY="$PWD/venv_umap/bin/python"
NGPU=$(nvidia-smi -L | wc -l)

# model spec: "kind:name"  (kind=hn -> hn_umap_atlas w/ read-threads; kind=web -> web_umap_atlas)
MODELS=(hn:qwen3_8b web:img_siglip2 web:img_clip web:img_nomic web:txt_bge_large web:txt_e5_large web:txt_jina_v3 web:txt_nomic)

run_one(){  # $1=phase(reduce|umap) $2=gpu $3=spec
  local ph=$1 gpu=$2 kind=${3%%:*} name=${3#*:} log
  log="umap_${kind}_${name}.log"
  local extra=""; [ "$ph" = reduce ] && extra="--reduce-only"
  if [ "$kind" = hn ]; then
    CUDA_VISIBLE_DEVICES=$gpu "$PY" umap/hn_umap_atlas.py --model "$name" --dims 2d,3d --read-threads 8 $extra >>"$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu "$PY" umap/web_umap_atlas.py --model "$name" --dims 2d,3d $extra >>"$log" 2>&1
  fi
}

for spec in "${MODELS[@]}"; do : > "umap_${spec%%:*}_${spec#*:}.log"; done

echo "=== PHASE 1: reduce (concurrent, 1/GPU) ==="
i=0; pids=()
for spec in "${MODELS[@]}"; do
  gpu=$((i % NGPU)); run_one reduce "$gpu" "$spec" & pids+=($!)
  i=$((i+1))
  # keep at most NGPU reduces in flight
  if [ $((i % NGPU)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== PHASE 1 done ==="

echo "=== PHASE 2: UMAP 2d+3d (1 job/GPU) ==="
i=0; pids=()
for spec in "${MODELS[@]}"; do
  gpu=$((i % NGPU)); run_one umap "$gpu" "$spec" & pids+=($!)
  i=$((i+1))
  if [ $((i % NGPU)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== ALL UMAP DONE ==="

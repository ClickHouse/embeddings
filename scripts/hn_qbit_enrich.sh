#!/usr/bin/env bash
# Add the QBit quantization columns to the 18 HackerNews embedding tables, mirroring
# mmcommons.emb_<model> (siglip2/clip/nomic). Each model's base `embedding` is QBit(BFloat16,dim)
# with stride == dim, so it CANNOT serve the dim-truncated cosineDistanceTransposed search
# (used_dims must be a multiple of the stride). We therefore create 4 stride-128 columns:
#   embedding_strided     QBit(BFloat16, dim, 128)  -- "BFloat16 | original"  (truncatable)
#   embedding_rotated     QBit(BFloat16, dim, 128)  -- "BFloat16 | rotated"   (randomHadamardTransform)
#   embedding_int         QBit(Int8,     dim, 128)  -- "Int8 | original"      (x*sqrt(dim) scaling)
#   embedding_rotated_int QBit(Int8,     dim, 128)  -- "Int8 | rotated"
# All HN models are unit-normalized (mean L2 ~= 1), so int8 uses the x*sqrt(dim) companding.
# HN dims are all multiples of 128 and randomHadamardTransform does NOT pad them (rdim == dim).
# Ordered small->large so the small models are searchable first. Idempotent (IF NOT EXISTS);
# re-running re-materialises. qwen3_8b is intentionally excluded (no UMAP tables; 4096-d/1.1TiB).
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -vi "unknown setting"; }
declare -A DIM=( [arctic_xs]=384 [bge_small]=384 [e5_small]=384 [granite_small]=384 [gte_small]=384 [minilm]=384 \
  [jina_small]=512 [arctic_m]=768 [bge_base]=768 [e5_base]=768 [embeddinggemma]=768 [gte_base]=768 [nomic]=768 \
  [kalm]=896 [bge_large]=1024 [e5_large]=1024 [jina_v3]=1024 [bow]=8192 )
ORDER="minilm arctic_xs bge_small e5_small granite_small gte_small jina_small nomic arctic_m bge_base e5_base embeddinggemma gte_base kalm bge_large e5_large jina_v3 bow"

wait_mut(){  # $1 = table name (no db)
  while :; do
    read n f <<< "$(CL "SELECT countIf(NOT is_done), countIf(NOT is_done AND latest_fail_reason!='') FROM system.mutations WHERE database='default' AND table='$1' FORMAT TSV")"
    if [ -n "${f:-}" ] && [ "$f" != "0" ]; then
      echo "MUT FAIL $1:"; CL "SELECT mutation_id, substring(latest_fail_reason,1,240) FROM system.mutations WHERE database='default' AND table='$1' AND NOT is_done AND latest_fail_reason!='' LIMIT 3"
      return 1
    fi
    [ "${n:-0}" = "0" ] && return 0
    echo "   pending mutations=$n  $(date +%H:%M:%S)"; sleep 30
  done
}

for m in $ORDER; do
  D=${DIM[$m]}; T=default.hackernews_embeddings_$m
  if [ "$(CL "SELECT count() FROM system.columns WHERE database='default' AND table='hackernews_embeddings_$m' AND name='embedding_rotated_int' FORMAT TSV")" = "1" ]; then
    echo "=== SKIP $m (already has embedding_rotated_int)  $(date +%T) ==="; continue
  fi
  echo "=== $m  dim=$D  $(date +%T) ==="
  echo "  pass A: embedding_rotated (Hadamard)"
  CL "ALTER TABLE $T ADD COLUMN IF NOT EXISTS embedding_rotated QBit(BFloat16,$D,128) DEFAULT randomHadamardTransform(CAST(embedding,'Array(BFloat16)'))"
  CL "ALTER TABLE $T MATERIALIZE COLUMN embedding_rotated SETTINGS mutations_sync=0"
  wait_mut hackernews_embeddings_$m || exit 1
  echo "  pass B: strided + int + rotated_int"
  CL "ALTER TABLE $T
      ADD COLUMN IF NOT EXISTS embedding_strided     QBit(BFloat16,$D,128) DEFAULT CAST(embedding,'Array(BFloat16)'),
      ADD COLUMN IF NOT EXISTS embedding_int         QBit(Int8,$D,128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt($D))), CAST(embedding,'Array(BFloat16)')) CODEC(NONE),
      ADD COLUMN IF NOT EXISTS embedding_rotated_int QBit(Int8,$D,128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt($D))), CAST(embedding_rotated,'Array(BFloat16)')) CODEC(NONE)"
  CL "ALTER TABLE $T MATERIALIZE COLUMN embedding_strided, MATERIALIZE COLUMN embedding_int, MATERIALIZE COLUMN embedding_rotated_int SETTINGS mutations_sync=0"
  wait_mut hackernews_embeddings_$m || exit 1
  echo "OK $m  $(date +%T)"
done
echo "=== HN QBIT ENRICH DONE  $(date +%T) ==="

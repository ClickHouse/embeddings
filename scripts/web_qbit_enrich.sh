#!/usr/bin/env bash
# Add the QBit quantization columns to the 7 WonderfulWeb embedding tables (web_emb_img_* + web_emb_txt_*),
# mirroring hn_qbit_enrich.sh / mmcommons.emb_<model>. Their base `embedding` is Array(BFloat16); we add 4
# stride-128 QBit columns so the Explorer's rep switch (Original/rotated x BFloat16/Int8, dim/bits sweep) works:
#   embedding_strided     QBit(BFloat16, dim, 128)  -- "BFloat16 | original" (truncatable)
#   embedding_rotated     QBit(BFloat16, dim, 128)  -- "BFloat16 | rotated"  (randomHadamardTransform)
#   embedding_int         QBit(Int8,     dim, 128)  -- "Int8 | original"     (x*sqrt(dim) companding)
#   embedding_rotated_int QBit(Int8,     dim, 128)  -- "Int8 | rotated"
# All 7 models are unit-normalized (avg L2 ~= 1.0) so int8 uses x*sqrt(dim). All dims are multiples of 128, so
# randomHadamardTransform does not pad (rdim == dim) and stride-128 truncation is clean. Ordered small->large.
# Idempotent (IF NOT EXISTS + skip if embedding_rotated_int already present); re-running re-materialises.
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -vi "unknown setting"; }
LOG=/tmp/web_qbit.log
declare -A DIM=( [web_emb_img_clip]=768 [web_emb_img_nomic]=768 [web_emb_txt_nomic]=768 \
  [web_emb_txt_bge_large]=1024 [web_emb_txt_e5_large]=1024 [web_emb_txt_jina_v3]=1024 [web_emb_img_siglip2]=1152 )
ORDER="web_emb_img_clip web_emb_img_nomic web_emb_txt_nomic web_emb_txt_bge_large web_emb_txt_e5_large web_emb_txt_jina_v3 web_emb_img_siglip2"

wait_mut(){  # $1 = table name (no db)
  while :; do
    read n f <<< "$(CL "SELECT countIf(NOT is_done), countIf(NOT is_done AND latest_fail_reason!='') FROM system.mutations WHERE database='default' AND table='$1' FORMAT TSV")"
    if [ -n "${f:-}" ] && [ "$f" != "0" ]; then
      echo "$(date +%T) MUT FAIL $1" >>"$LOG"
      CL "SELECT mutation_id, substring(latest_fail_reason,1,240) FROM system.mutations WHERE database='default' AND table='$1' AND NOT is_done AND latest_fail_reason!='' LIMIT 3" >>"$LOG"
      return 1
    fi
    [ "${n:-0}" = "0" ] && return 0
    echo "$(date +%T)   $1 pending mutations=$n" >>"$LOG"; sleep 30
  done
}

[ -z "${CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD:-}" ] && { echo "$(date +%T) ABORT: dest pw empty" >>"$LOG"; exit 1; }
echo "=== web_qbit start $(date) ===" >>"$LOG"
for T in $ORDER; do
  D=${DIM[$T]}
  if [ "$(CL "SELECT count() FROM system.columns WHERE database='default' AND table='$T' AND name='embedding_rotated_int' FORMAT TSV")" = "1" ]; then
    echo "$(date +%T) SKIP $T (already has embedding_rotated_int)" >>"$LOG"; continue
  fi
  echo "$(date +%T) === $T dim=$D pass A: embedding_rotated ===" >>"$LOG"
  CL "ALTER TABLE default.$T ADD COLUMN IF NOT EXISTS embedding_rotated QBit(BFloat16,$D,128) DEFAULT randomHadamardTransform(CAST(embedding,'Array(BFloat16)'))"
  CL "ALTER TABLE default.$T MATERIALIZE COLUMN embedding_rotated SETTINGS mutations_sync=0"
  wait_mut "$T" || exit 1
  echo "$(date +%T) === $T pass B: strided + int + rotated_int ===" >>"$LOG"
  CL "ALTER TABLE default.$T
      ADD COLUMN IF NOT EXISTS embedding_strided     QBit(BFloat16,$D,128) DEFAULT CAST(embedding,'Array(BFloat16)'),
      ADD COLUMN IF NOT EXISTS embedding_int         QBit(Int8,$D,128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt($D))), CAST(embedding,'Array(BFloat16)')) CODEC(NONE),
      ADD COLUMN IF NOT EXISTS embedding_rotated_int QBit(Int8,$D,128) DEFAULT arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt($D))), CAST(embedding_rotated,'Array(BFloat16)')) CODEC(NONE)"
  CL "ALTER TABLE default.$T MATERIALIZE COLUMN embedding_strided, MATERIALIZE COLUMN embedding_int, MATERIALIZE COLUMN embedding_rotated_int SETTINGS mutations_sync=0"
  wait_mut "$T" || exit 1
  echo "$(date +%T) OK $T" >>"$LOG"
done
echo "=== web_qbit DONE $(date) ===" >>"$LOG"

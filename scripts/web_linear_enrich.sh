#!/usr/bin/env bash
# Linear projection for the 7 WonderfulWeb embedding tables: x,y,z = mapped sums of the 3 equal thirds of the
# Hadamard-rotated embedding (mirrors hn_linear_enrich.sh / mmcommons.emb_siglip2; z doubles as colour/hue).
# Requires embedding_rotated materialised (scripts/web_qbit_enrich.sh — done). Keyed by url. Idempotent
# (skip if x exists). Emits per-model `lq` = 95th-pct linear per-pixel density (the LOWER quantile the frontend
# uses for linear brightness — linear is sparser/peakier than UMAP, so q99/uq99 renders it too dim).
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -vi "unknown setting"; }
LOG=/tmp/web_linear.log
declare -A DIM=( [web_emb_img_clip]=768 [web_emb_img_nomic]=768 [web_emb_txt_nomic]=768 \
  [web_emb_txt_bge_large]=1024 [web_emb_txt_e5_large]=1024 [web_emb_txt_jina_v3]=1024 [web_emb_img_siglip2]=1152 )
ORDER="web_emb_img_clip web_emb_img_nomic web_emb_txt_nomic web_emb_txt_bge_large web_emb_txt_e5_large web_emb_txt_jina_v3 web_emb_img_siglip2"

wait_mut(){  # $1 = table (no db)
  while :; do
    read n f <<< "$(CL "SELECT countIf(NOT is_done), countIf(NOT is_done AND latest_fail_reason!='') FROM system.mutations WHERE database='default' AND table='$1' FORMAT TSV")"
    if [ -n "${f:-}" ] && [ "$f" != "0" ]; then
      echo "$(date +%T) MUT FAIL $1" >>"$LOG"
      CL "SELECT mutation_id, substring(latest_fail_reason,1,240) FROM system.mutations WHERE database='default' AND table='$1' AND NOT is_done AND latest_fail_reason!='' LIMIT 3" >>"$LOG"; return 1
    fi
    [ "${n:-0}" = "0" ] && return 0
    sleep 30
  done
}

[ -z "${CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD:-}" ] && { echo "$(date +%T) ABORT: pw empty" >>"$LOG"; exit 1; }
echo "=== WEB LINEAR ENRICH start $(date) ===" >>"$LOG"
for T in $ORDER; do
  D=${DIM[$T]}; k=$((D/3))
  if [ "$(CL "SELECT count() FROM system.columns WHERE database='default' AND table='$T' AND name='x' FORMAT TSV")" = "1" ]; then
    echo "$(date +%T) SKIP $T (x exists)" >>"$LOG"; continue
  fi
  echo "$(date +%T) === $T dim=$D third=$k : measure third-sum ranges ===" >>"$LOG"
  read n1 x1 n2 x2 n3 x3 <<< "$(CL "WITH CAST(embedding_rotated,'Array(Float32)') AS v
      SELECT round(min(arraySum(arraySlice(v,1,$k))),5), round(max(arraySum(arraySlice(v,1,$k))),5),
             round(min(arraySum(arraySlice(v,$((k+1)),$k))),5), round(max(arraySum(arraySlice(v,$((k+1)),$k))),5),
             round(min(arraySum(arraySlice(v,$((2*k+1))))),5), round(max(arraySum(arraySlice(v,$((2*k+1))))),5)
      FROM default.$T FORMAT TSV")"
  echo "$(date +%T)   ranges: x[$n1,$x1] y[$n2,$x2] z[$n3,$x3]" >>"$LOG"
  CL "ALTER TABLE default.$T
      ADD COLUMN IF NOT EXISTS x UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),1,$k))-($n1))/(($x1)-($n1)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS y UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),$((k+1)),$k))-($n2))/(($x2)-($n2)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS z UInt16 MATERIALIZED toUInt16(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),$((2*k+1))))-($n3))/(($x3)-($n3)),0.,1.)*65535))"
  CL "ALTER TABLE default.$T MATERIALIZE COLUMN x, MATERIALIZE COLUMN y, MATERIALIZE COLUMN z SETTINGS mutations_sync=0"
  wait_mut "$T" || exit 1
  CL "ALTER TABLE default.$T ADD PROJECTION IF NOT EXISTS proj_xy (SELECT x,y,z,url ORDER BY mortonEncode(x,y))"
  CL "ALTER TABLE default.$T MATERIALIZE PROJECTION proj_xy SETTINGS mutations_sync=0"
  wait_mut "$T" || exit 1
  lq=$(CL "SELECT round(quantile(0.95)(cnt)) FROM (SELECT intDiv(x,4194304) gx, intDiv(y,4194304) gy, count() cnt FROM default.$T GROUP BY gx,gy) FORMAT TSV")
  echo "$(date +%T) OK $T | lq=$lq | x[$n1,$x1] y[$n2,$x2] z[$n3,$x3]" >>"$LOG"
done
echo "=== WEB LINEAR ENRICH DONE $(date) ===" >>"$LOG"

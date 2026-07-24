#!/usr/bin/env bash
# Linear projection for the HackerNews embedding tables: x,y,z = mapped sums of the 3 equal thirds of the
# Hadamard-rotated embedding (mirrors mmcommons.emb_siglip2's x/y/z). z doubles as the point colour (hue).
# Requires embedding_rotated to be MATERIALISED first (scripts/hn_qbit_enrich.sh, pass A); models whose
# embedding_rotated isn't ready yet are skipped with a note — just re-run later. Idempotent (IF NOT EXISTS).
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -vi "unknown setting"; }
declare -A DIM=( [minilm]=384 [arctic_xs]=384 [bge_small]=384 [e5_small]=384 [granite_small]=384 [gte_small]=384 \
  [jina_small]=512 [nomic]=768 [arctic_m]=768 [bge_base]=768 [e5_base]=768 [embeddinggemma]=768 [gte_base]=768 \
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
  D=${DIM[$m]}; T=default.hackernews_embeddings_$m; k=$((D/3))
  if [ "$(CL "SELECT count() FROM system.columns WHERE database='default' AND table='hackernews_embeddings_$m' AND name='x' FORMAT TSV")" = "1" ]; then
    echo "SKIP $m (x already exists)  $(date +%T)"; continue
  fi
  rot=$(CL "SELECT (SELECT count() FROM system.parts_columns WHERE database='default' AND table='hackernews_embeddings_$m' AND active AND column='embedding_rotated') >= (SELECT count() FROM system.parts WHERE database='default' AND table='hackernews_embeddings_$m' AND active) AND (SELECT count() FROM system.parts WHERE database='default' AND table='hackernews_embeddings_$m' AND active) > 0 FORMAT TSV")
  if [ "$rot" != "1" ]; then echo "PENDING $m (embedding_rotated not materialised yet — re-run later)  $(date +%T)"; continue; fi
  echo "=== $m  dim=$D  third=$k  $(date +%T) ==="
  # per-third sum ranges over the rotated embedding -> offset/scale to map into [0,2^32) (x,y) and [0,65535] (z=colour)
  read n1 x1 n2 x2 n3 x3 <<< "$(CL "WITH CAST(embedding_rotated,'Array(Float32)') AS v
      SELECT round(min(arraySum(arraySlice(v,1,$k))),5), round(max(arraySum(arraySlice(v,1,$k))),5),
             round(min(arraySum(arraySlice(v,$((k+1)),$k))),5), round(max(arraySum(arraySlice(v,$((k+1)),$k))),5),
             round(min(arraySum(arraySlice(v,$((2*k+1))))),5), round(max(arraySum(arraySlice(v,$((2*k+1))))),5)
      FROM $T FORMAT TSV")"
  CL "ALTER TABLE $T
      ADD COLUMN IF NOT EXISTS x UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),1,$k))-($n1))/(($x1)-($n1)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS y UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),$((k+1)),$k))-($n2))/(($x2)-($n2)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS z UInt16 MATERIALIZED toUInt16(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),$((2*k+1))))-($n3))/(($x3)-($n3)),0.,1.)*65535))"
  CL "ALTER TABLE $T MATERIALIZE COLUMN x, MATERIALIZE COLUMN y, MATERIALIZE COLUMN z SETTINGS mutations_sync=0"
  wait_mut hackernews_embeddings_$m || exit 1
  CL "ALTER TABLE $T ADD PROJECTION IF NOT EXISTS proj_xy (SELECT x,y,z,id ORDER BY mortonEncode(x,y))"
  CL "ALTER TABLE $T MATERIALIZE PROJECTION proj_xy SETTINGS mutations_sync=0"
  wait_mut hackernews_embeddings_$m || exit 1
  q99=$(CL "SELECT round(quantile(0.99)(cnt)) FROM (SELECT intDiv(x,4194304) gx, intDiv(y,4194304) gy, count() cnt FROM $T GROUP BY gx,gy) FORMAT TSV")
  echo "OK $m  linear q99=$q99  $(date +%T)"
done
echo "=== HN LINEAR ENRICH DONE  $(date +%T) ==="

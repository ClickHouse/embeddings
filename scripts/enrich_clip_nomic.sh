#!/usr/bin/env bash
# Enrich mmcommons.emb_clip and emb_nomic (768-dim) with the same columns/projection as siglip2.
# rotated dim = 768 (randomHadamardTransform does NOT pad 768); x/y/z from 3x256 slices with per-model measured constants.
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -vi "unknown setting"; }
wait_mut(){
  while :; do
    read n f <<< "$(CL "SELECT countIf(NOT is_done), countIf(NOT is_done AND latest_fail_reason != '') FROM system.mutations WHERE database='mmcommons' AND table IN ('emb_clip','emb_nomic') FORMAT TSV")"
    if [ -n "$f" ] && [ "$f" != "0" ]; then
      echo "MUTATION FAIL:"; CL "SELECT table, mutation_id, substring(latest_fail_reason,1,220) FROM system.mutations WHERE database='mmcommons' AND table IN ('emb_clip','emb_nomic') AND NOT is_done AND latest_fail_reason != '' LIMIT 4"
      return 1
    fi
    [ "$n" = "0" ] && return 0
    echo "  pending mutations=$n  $(date +%H:%M:%S)"; sleep 60
  done
}
echo "=== PASS 1: embedding_rotated (Hadamard, 768) $(date +%H:%M:%S) ==="
CL "ALTER TABLE mmcommons.emb_clip  ADD COLUMN IF NOT EXISTS embedding_rotated QBit(BFloat16,768,128) DEFAULT randomHadamardTransform(CAST(embedding,'Array(BFloat16)'))"
CL "ALTER TABLE mmcommons.emb_nomic ADD COLUMN IF NOT EXISTS embedding_rotated QBit(BFloat16,768,128) DEFAULT randomHadamardTransform(CAST(embedding,'Array(BFloat16)'))"
CL "ALTER TABLE mmcommons.emb_clip  MATERIALIZE COLUMN embedding_rotated SETTINGS mutations_sync=0"
CL "ALTER TABLE mmcommons.emb_nomic MATERIALIZE COLUMN embedding_rotated SETTINGS mutations_sync=0"
wait_mut || exit 1

echo "=== PASS 2: strided/int/rotated_int + x/y/z $(date +%H:%M:%S) ==="
CL "ALTER TABLE mmcommons.emb_clip
  ADD COLUMN IF NOT EXISTS embedding_strided     QBit(BFloat16,768,128) DEFAULT CAST(embedding,'Array(BFloat16)'),
  ADD COLUMN IF NOT EXISTS embedding_int         QBit(Int8,768,128) DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(embedding,'Array(BFloat16)')) CODEC(NONE),
  ADD COLUMN IF NOT EXISTS embedding_rotated_int QBit(Int8,768,128) DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(embedding_rotated,'Array(BFloat16)')) CODEC(NONE),
  ADD COLUMN IF NOT EXISTS x UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),  1,256)) + 1.0025)/1.8655,0.,1.)*4294967295)),
  ADD COLUMN IF NOT EXISTS y UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),257,256)) + 0.8934)/1.72,0.,1.)*4294967295)),
  ADD COLUMN IF NOT EXISTS z UInt16 MATERIALIZED toUInt16(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),513,256)) - 0.2332)/0.5748,0.,1.)*65535))"
CL "ALTER TABLE mmcommons.emb_nomic
  ADD COLUMN IF NOT EXISTS embedding_strided     QBit(BFloat16,768,128) DEFAULT CAST(embedding,'Array(BFloat16)'),
  ADD COLUMN IF NOT EXISTS embedding_int         QBit(Int8,768,128) DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(embedding,'Array(BFloat16)')) CODEC(NONE),
  ADD COLUMN IF NOT EXISTS embedding_rotated_int QBit(Int8,768,128) DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(embedding_rotated,'Array(BFloat16)')) CODEC(NONE),
  ADD COLUMN IF NOT EXISTS x UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),  1,256)) + 0.3414)/1.54,0.,1.)*4294967295)),
  ADD COLUMN IF NOT EXISTS y UInt32 MATERIALIZED toUInt32(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),257,256)) + 1.5506)/1.4005,0.,1.)*4294967295)),
  ADD COLUMN IF NOT EXISTS z UInt16 MATERIALIZED toUInt16(round(clamp((arraySum(arraySlice(CAST(embedding_rotated,'Array(Float32)'),513,256)) + 0.527)/0.3644,0.,1.)*65535))"
for T in emb_clip emb_nomic; do
  CL "ALTER TABLE mmcommons.$T MATERIALIZE COLUMN embedding_strided, MATERIALIZE COLUMN embedding_int, MATERIALIZE COLUMN embedding_rotated_int, MATERIALIZE COLUMN x, MATERIALIZE COLUMN y, MATERIALIZE COLUMN z SETTINGS mutations_sync=0"
done
wait_mut || exit 1

echo "=== PASS 3: proj_xy projection $(date +%H:%M:%S) ==="
CL "ALTER TABLE mmcommons.emb_clip  ADD PROJECTION IF NOT EXISTS proj_xy (SELECT x,y,z,md5 ORDER BY mortonEncode(x,y))"
CL "ALTER TABLE mmcommons.emb_nomic ADD PROJECTION IF NOT EXISTS proj_xy (SELECT x,y,z,md5 ORDER BY mortonEncode(x,y))"
CL "ALTER TABLE mmcommons.emb_clip  MATERIALIZE PROJECTION proj_xy SETTINGS mutations_sync=0"
CL "ALTER TABLE mmcommons.emb_nomic MATERIALIZE PROJECTION proj_xy SETTINGS mutations_sync=0"
wait_mut || exit 1

echo "=== VERIFY $(date +%H:%M:%S) ==="
CL "SELECT table, arraySort(groupArray(name)) FROM system.columns WHERE database='mmcommons' AND table IN ('emb_clip','emb_nomic') GROUP BY table FORMAT Vertical"
CL "SELECT table, countIf(has(projections,'proj_xy')) AS parts_with_proj, count() AS parts FROM system.parts WHERE database='mmcommons' AND table IN ('emb_clip','emb_nomic') AND active GROUP BY table FORMAT PrettyCompact"
echo "ENRICH DONE $(date +%H:%M:%S)"

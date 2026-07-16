#!/usr/bin/env bash
# Recreate embedding_int / embedding_rotated_int with sqrt(N) scaling before quantization
# (embeddings are unit-normalized -> coords ~N(0,1/sqrt(N)); *sqrt(N) -> ~N(0,1), the quantizer's range).
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -vi "unknown setting"; }
wait_mut(){
  while :; do
    n=$(CL "SELECT countIf(NOT is_done) FROM system.mutations WHERE database='mmcommons' AND table IN ('emb_siglip2','emb_clip','emb_nomic')")
    hard=$(CL "SELECT countIf(NOT is_done AND match(latest_fail_reason, 'Code: (190|47|53|43|36|41)')) FROM system.mutations WHERE database='mmcommons' AND table IN ('emb_siglip2','emb_clip','emb_nomic')")
    if [ -n "$hard" ] && [ "$hard" != "0" ]; then
      echo "HARD FAIL:"; CL "SELECT table, mutation_id, substring(latest_fail_reason,1,200) FROM system.mutations WHERE database='mmcommons' AND table IN ('emb_siglip2','emb_clip','emb_nomic') AND NOT is_done AND latest_fail_reason != '' LIMIT 6"; return 1
    fi
    [ "$n" = "0" ] && return 0
    echo "  pending=$n  $(date +%H:%M:%S)"; sleep 60
  done
}
si_int="arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(1152))), CAST(embedding,'Array(BFloat16)'))"
si_rot="arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(2048))), CAST(embedding_rotated,'Array(BFloat16)'))"
sm_int="arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(768))),  CAST(embedding,'Array(BFloat16)'))"
sm_rot="arrayMap(x -> quantizeBFloat16ToInt8(toBFloat16(x * sqrt(768))),  CAST(embedding_rotated,'Array(BFloat16)'))"

echo "=== MODIFY column definitions (scaled) $(date +%H:%M:%S) ==="
CL "ALTER TABLE mmcommons.emb_siglip2
  MODIFY COLUMN embedding_int         QBit(Int8,1152,128) DEFAULT $si_int CODEC(NONE),
  MODIFY COLUMN embedding_rotated_int QBit(Int8,2048,128) DEFAULT $si_rot CODEC(NONE)"
CL "ALTER TABLE mmcommons.emb_clip
  MODIFY COLUMN embedding_int         QBit(Int8,768,128) DEFAULT $sm_int CODEC(NONE),
  MODIFY COLUMN embedding_rotated_int QBit(Int8,768,128) DEFAULT $sm_rot CODEC(NONE)"
CL "ALTER TABLE mmcommons.emb_nomic
  MODIFY COLUMN embedding_int         QBit(Int8,768,128) DEFAULT $sm_int CODEC(NONE),
  MODIFY COLUMN embedding_rotated_int QBit(Int8,768,128) DEFAULT $sm_rot CODEC(NONE)"
echo "definitions now:"
CL "SELECT table, name, substring(default_expression,1,60) FROM system.columns WHERE database='mmcommons' AND table IN ('emb_siglip2','emb_clip','emb_nomic') AND name IN ('embedding_int','embedding_rotated_int') ORDER BY table, name FORMAT TSV"

echo "=== MATERIALIZE (re-quantize) $(date +%H:%M:%S) ==="
for T in emb_siglip2 emb_clip emb_nomic; do
  CL "ALTER TABLE mmcommons.$T MATERIALIZE COLUMN embedding_int, MATERIALIZE COLUMN embedding_rotated_int SETTINGS mutations_sync=0"
done
wait_mut || exit 1

echo "=== VERIFY new Int8 ranges (expect wide, std ~50-60, |q| up to ~127) $(date +%H:%M:%S) ==="
for T in emb_siglip2 emb_clip emb_nomic; do
  CL "SELECT '$T' t, min(q) qmin, max(q) qmax, round(stddevPop(q),1) qstd FROM (SELECT arrayJoin(CAST(embedding_int,'Array(Int8)')) q FROM mmcommons.$T LIMIT 500) FORMAT TSV"
done
echo "RESCALE DONE $(date +%H:%M:%S)"

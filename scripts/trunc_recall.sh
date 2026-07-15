#!/bin/bash
# Average recall@100 vs true cosine, per top-N-bit truncation, original vs rotated int8, over several queries.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -v "Unknown settings"; }
# 12 pseudo-random query photo_ids (early-terminating sample)
QIDS=$(CC "SELECT photo_id FROM mmcommons.search_nomic WHERE cityHash64(photo_id)%1000=0 LIMIT 12")
echo "queries: $QIDS"
: > /tmp/trunc_recall.tsv
for Q in $QIDS; do
  CC "
  WITH
    (SELECT embedding_int::Array(Int8) FROM mmcommons.search_nomic WHERE photo_id=$Q) AS qi,
    (SELECT embedding_rotated_int::Array(Int8) FROM mmcommons.search_nomic WHERE photo_id=$Q) AS qri,
    (SELECT embedding::Array(BFloat16) FROM mmcommons.search_nomic WHERE photo_id=$Q) AS qtrue
  SELECT N,
    length(arrayIntersect(true_top, orig_top))/100. AS r_orig,
    length(arrayIntersect(true_top, rot_top))/100.  AS r_rot
  FROM (
    SELECT N,
      arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_true, photo_id)))),1,100) AS true_top,
      arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_orig, photo_id)))),1,100) AS orig_top,
      arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_rot, photo_id)))),1,100) AS rot_top
    FROM (
      SELECT N, mask, photo_id, d_true,
        cosineDistance(arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,mask))),di),  arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,mask))),qi))  AS d_orig,
        cosineDistance(arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,mask))),dri), arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,mask))),qri)) AS d_rot
      FROM (SELECT photo_id, embedding_int::Array(Int8) AS di, embedding_rotated_int::Array(Int8) AS dri,
                   cosineDistance(embedding::Array(BFloat16),qtrue) AS d_true
            FROM mmcommons.search_nomic LIMIT 10000)
      ARRAY JOIN [1,2,3,4,5,6,7,8] AS N, [-128,-64,-32,-16,-8,-4,-2,-1] AS mask
    ) GROUP BY N
  ) ORDER BY N FORMAT TSV" >> /tmp/trunc_recall.tsv
  echo "  done query $Q"
done
echo "=== AVERAGE recall@100 over all queries (N  recall_orig  recall_rot) ==="
awk -F'\t' '{o[$1]+=$2; r[$1]+=$3; n[$1]++} END{for(k=8;k>=1;k--) printf "%d\t%.3f\t%.3f\n", k, o[k]/n[k], r[k]/n[k]}' /tmp/trunc_recall.tsv
#!/bin/bash
# Apples-to-apples recall@100 vs true cosine at N=8..1 bits:
#   int8 (top-N-bit truncation) vs BFloat16 QBit (cosineDistanceTransposed), original vs rotated.
# Fixed deterministic pool + fixed query set so all four are comparable.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -v "Unknown settings"; }
# fixed ~9k-row pool (hash-deterministic set) with all 4 embedding variants
CC "DROP TABLE IF EXISTS mmcommons.pool_tmp"
CC "CREATE TABLE mmcommons.pool_tmp (
      photo_id UInt64,
      embedding QBit(BFloat16,768,16),
      embedding_rotated QBit(BFloat16,768,16),
      embedding_int QBit(Int8,768,16),
      embedding_rotated_int QBit(Int8,768,16)
    ) ENGINE=MergeTree ORDER BY photo_id"
CC "INSERT INTO mmcommons.pool_tmp
    SELECT photo_id, embedding, embedding_rotated, embedding_int, embedding_rotated_int
    FROM mmcommons.search_nomic WHERE cityHash64(photo_id)%11000=0"
POOL=$(CC "SELECT count() FROM mmcommons.pool_tmp")
echo "pool rows: $POOL"
[ "$POOL" -gt 1000 ] 2>/dev/null || { echo "pool creation failed; aborting"; exit 1; }
QIDS=$(CC "SELECT photo_id FROM mmcommons.pool_tmp ORDER BY photo_id LIMIT 12")
: > /tmp/recall_compare.tsv
for Q in $QIDS; do
  for N in 8 7 6 5 4 3 2 1; do
    MASK=$(( - (1 << (8-N)) ))
    CC "
    WITH
      (SELECT embedding::Array(BFloat16) FROM mmcommons.pool_tmp WHERE photo_id=$Q) AS qref,
      (SELECT embedding_rotated::Array(BFloat16) FROM mmcommons.pool_tmp WHERE photo_id=$Q) AS qrref,
      (SELECT embedding_int::Array(Int8) FROM mmcommons.pool_tmp WHERE photo_id=$Q) AS qi,
      (SELECT embedding_rotated_int::Array(Int8) FROM mmcommons.pool_tmp WHERE photo_id=$Q) AS qri
    SELECT $N AS N,
      length(arrayIntersect(tt,i8o))/100. AS r_i8o,
      length(arrayIntersect(tt,i8r))/100. AS r_i8r,
      length(arrayIntersect(tt,bfo))/100. AS r_bfo,
      length(arrayIntersect(tt,bfr))/100. AS r_bfr
    FROM (
      SELECT
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_true,photo_id)))),1,100) AS tt,
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_i8o, photo_id)))),1,100) AS i8o,
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_i8r, photo_id)))),1,100) AS i8r,
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_bfo, photo_id)))),1,100) AS bfo,
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_bfr, photo_id)))),1,100) AS bfr
      FROM (
        SELECT photo_id,
          cosineDistance(embedding::Array(BFloat16), qref) AS d_true,
          cosineDistance(arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,$MASK))),embedding_int::Array(Int8)),        arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,$MASK))),qi))  AS d_i8o,
          cosineDistance(arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,$MASK))),embedding_rotated_int::Array(Int8)),arrayMap(c->dequantizeInt8ToBFloat16(toInt8(bitAnd(c,$MASK))),qri)) AS d_i8r,
          cosineDistanceTransposed(embedding, qref, $N)          AS d_bfo,
          cosineDistanceTransposed(embedding_rotated, qrref, $N) AS d_bfr
        FROM mmcommons.pool_tmp
      )
    ) FORMAT TSV" >> /tmp/recall_compare.tsv
  done
  echo "  done query $Q"
done
CC "DROP TABLE IF EXISTS mmcommons.pool_tmp"
echo "=== AVG recall@100  (N  int8_orig  int8_rot  bf16_orig  bf16_rot) ==="
awk -F'\t' '{a[$1]+=$2;b[$1]+=$3;c[$1]+=$4;d[$1]+=$5;n[$1]++} END{for(k=8;k>=1;k--) printf "%d\t%.3f\t%.3f\t%.3f\t%.3f\n",k,a[k]/n[k],b[k]/n[k],c[k]/n[k],d[k]/n[k]}' /tmp/recall_compare.tsv
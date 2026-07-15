#!/bin/bash
# Recall@100 vs true cosine using QBit BFloat16 transposed distance at N=8..1 bits,
# original vs rotated, averaged over the same 12 queries / 10k pool as the int8 test.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "$1" 2>&1 | grep -v "Unknown settings"; }
QIDS=$(CC "SELECT photo_id FROM mmcommons.search_nomic WHERE cityHash64(photo_id)%1000=0 LIMIT 12")
echo "queries: $(echo $QIDS)"
: > /tmp/trunc_recall_bf16.tsv
for Q in $QIDS; do
  for N in 8 7 6 5 4 3 2 1; do
    CC "
    WITH
      (SELECT embedding::Array(BFloat16) FROM mmcommons.search_nomic WHERE photo_id=$Q) AS qref,
      (SELECT embedding_rotated::Array(BFloat16) FROM mmcommons.search_nomic WHERE photo_id=$Q) AS qrref
    SELECT $N AS N,
      length(arrayIntersect(true_top, orig_top))/100. AS r_orig,
      length(arrayIntersect(true_top, rot_top))/100.  AS r_rot
    FROM (
      SELECT
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_true, photo_id)))),1,100) AS true_top,
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_orig, photo_id)))),1,100) AS orig_top,
        arraySlice(arrayMap(x->x.2, arraySort(x->x.1, groupArray((d_rot,  photo_id)))),1,100) AS rot_top
      FROM (
        SELECT photo_id,
          cosineDistance(embedding::Array(BFloat16), qref)       AS d_true,
          cosineDistanceTransposed(embedding, qref, $N)          AS d_orig,
          cosineDistanceTransposed(embedding_rotated, qrref, $N) AS d_rot
        FROM mmcommons.search_nomic LIMIT 10000
      )
    ) FORMAT TSV" >> /tmp/trunc_recall_bf16.tsv
  done
  echo "  done query $Q"
done
echo "=== AVERAGE recall@100, QBit BFloat16 transposed (N  bf16_orig  bf16_rot) ==="
awk -F'\t' '{o[$1]+=$2; r[$1]+=$3; n[$1]++} END{for(k=8;k>=1;k--) printf "%d\t%.3f\t%.3f\n", k, o[k]/n[k], r[k]/n[k]}' /tmp/trunc_recall_bf16.tsv
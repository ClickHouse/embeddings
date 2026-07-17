#!/usr/bin/env bash
# Build the bench.* tables that scripts/recall_bench.py consumes:
#   samp_<ds> : 100k-row random sample (first 100k by md5 -> random w.r.t. embeddings, PK-pruned so no full scan)
#   q_<ds>    : 20 random query points (q_idx, q_md5, ref_orig, ref_rot)
#   gt_<ds>   : exact top-100 ground truth per query (gt10, gt100) by full-precision cosine
# MergeTree (not Memory) so the tables are visible across ClickHouse Cloud's load-balanced replicas.
set -e
CH=(clickhouse client --allow_experimental_qbit_type 1 --max_execution_time 300)   # adjust connection flags
q(){ "${CH[@]}" --query "$1" 2>&1 | grep -vi "unknown setting"; }
q "CREATE DATABASE IF NOT EXISTS bench"

# args: table  rotated-QBit  strided-QBit  int-QBit  rotated_int-QBit
setup(){ T=$1; ROT=$2; STR=$3; INT=$4; RINT=$5
  cut=$("${CH[@]}" --query "SELECT md5 FROM mmcommons.$T ORDER BY md5 LIMIT 1 OFFSET 100000" 2>/dev/null)
  q "DROP TABLE IF EXISTS bench.samp_$T"; q "DROP TABLE IF EXISTS bench.q_$T"; q "DROP TABLE IF EXISTS bench.gt_$T"

  q "CREATE TABLE bench.samp_$T (md5 String, embedding Array(BFloat16),
       embedding_rotated $ROT, embedding_strided $STR, embedding_int $INT, embedding_rotated_int $RINT)
     ENGINE = MergeTree ORDER BY md5"
  q "INSERT INTO bench.samp_$T
       SELECT md5, embedding, embedding_rotated, embedding_strided, embedding_int, embedding_rotated_int
       FROM mmcommons.$T WHERE md5 < '$cut'"

  q "CREATE TABLE bench.q_$T (q_idx UInt32, q_md5 String, ref_orig Array(Float32), ref_rot Array(Float32))
     ENGINE = MergeTree ORDER BY q_idx"
  q "INSERT INTO bench.q_$T
       SELECT rowNumberInAllBlocks(), md5, CAST(embedding,'Array(Float32)'), CAST(embedding_rotated,'Array(Float32)')
       FROM (SELECT md5, embedding, embedding_rotated FROM bench.samp_$T ORDER BY cityHash64(md5) LIMIT 20)"

  q "CREATE TABLE bench.gt_$T (q_idx UInt32, gt10 Array(String), gt100 Array(String))
     ENGINE = MergeTree ORDER BY q_idx"
  q "INSERT INTO bench.gt_$T
       SELECT q_idx,
              arraySlice(arrayMap(t->t.2, arraySort(t->t.1, groupArray((dist,md5)))),1,10),
              arrayMap(t->t.2, arraySort(t->t.1, groupArray((dist,md5))))
       FROM (SELECT q.q_idx q_idx, s.md5 md5, cosineDistance(CAST(s.embedding,'Array(Float32)'), q.ref_orig) dist
             FROM bench.samp_$T s CROSS JOIN bench.q_$T q WHERE s.md5 != q.q_md5
             ORDER BY q_idx, dist ASC LIMIT 100 BY q_idx) GROUP BY q_idx"
  echo "$T: samp=$("${CH[@]}" --query "SELECT count() FROM bench.samp_$T" 2>/dev/null) q=$("${CH[@]}" --query "SELECT count() FROM bench.q_$T" 2>/dev/null) gt=$("${CH[@]}" --query "SELECT count() FROM bench.gt_$T" 2>/dev/null)"
}

setup emb_siglip2 'QBit(BFloat16,2048,128)' 'QBit(BFloat16,1152,128)' 'QBit(Int8,1152,128)' 'QBit(Int8,2048,128)'
setup emb_clip    'QBit(BFloat16,768,128)'  'QBit(BFloat16,768,128)'  'QBit(Int8,768,128)'  'QBit(Int8,768,128)'
setup emb_nomic   'QBit(BFloat16,768,128)'  'QBit(BFloat16,768,128)'  'QBit(Int8,768,128)'  'QBit(Int8,768,128)'
echo "setup done — now run: python3 scripts/recall_bench.py"

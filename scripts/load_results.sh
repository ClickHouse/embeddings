#!/bin/bash
# Load each results/<slug>.parquet into a per-model MergeTree table emb_<slug> (id, embedding)
# in the main clickhouse-local path, then verify.
cd /home/ubuntu/embeddings
CH="./clickhouse-new local --path ."

while IFS=$'\t' read -r slug trunc model; do
  out="results/${slug}.parquet"
  [ -f "$out" ] || { echo "MISSING $slug"; continue; }
  n=$(./clickhouse-new local --query "SELECT count() FROM file('$out')" 2>/dev/null)
  [ "$n" = "10000" ] || { echo "INCOMPLETE $slug rows=$n (skipping load)"; continue; }
  $CH --query "
    CREATE OR REPLACE TABLE emb_${slug} (id UInt32, embedding Array(Float32)) ENGINE = MergeTree ORDER BY id;
    INSERT INTO emb_${slug} SELECT id, embedding FROM file('${out}');
  " 2>&1 | tail -1
  rows=$($CH --query "SELECT count() FROM emb_${slug}")
  dim=$($CH --query "SELECT length(embedding) FROM emb_${slug} LIMIT 1")
  echo "LOADED emb_${slug} rows=$rows dim=$dim"
done < jobs.tsv
echo "ALL_LOADED"

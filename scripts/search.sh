cd /home/ubuntu/embeddings
./clickhouse-new local --path . --allow_experimental_qbit_type 1 --query "
WITH (SELECT q::Array(BFloat16) FROM file('qvec.parquet')) AS qv
SELECT c.id AS id, c.dist AS dist, h.doc AS doc, (h.doc ILIKE '%clickhouse%') AS has_ch
FROM (
  SELECT id, cosineDistanceTransposed(embedding, qv, 16) AS dist
  FROM emb_full_qwen ORDER BY dist ASC LIMIT 200
) c
JOIN file('hn_in/shard_*.parquet', Parquet, 'id UInt32, doc String') h ON c.id = h.id
INTO OUTFILE 'cand.parquet' TRUNCATE FORMAT Parquet
SETTINGS max_threads=64, optimize_qbit_distance_function_reads=1"
echo "SEARCH_RC=$?"

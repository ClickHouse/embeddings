cd /home/ubuntu/embeddings
echo "--- full scan, max_threads=64 ---"
./clickhouse-new local --path . --allow_experimental_qbit_type 1 --send_logs_level=trace --query "
WITH (SELECT q::Array(BFloat16) FROM file('qvec.parquet')) AS qv
SELECT id, cosineDistanceTransposed(embedding, qv, 16) AS dist
FROM emb_full_qwen ORDER BY dist ASC LIMIT 5
SETTINGS max_threads=64 FORMAT Null" 2>&1 | grep -iE 'Peak memory usage \(for query\)' | tail -1

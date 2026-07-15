cd /home/ubuntu/embeddings
./clickhouse-new local --path . --allow_experimental_qbit_type 1 --query "
INSERT INTO emb_full_qwen
  SELECT id, embedding
  FROM file('results/qwen_full/*.parquet', Parquet, 'id UInt32, embedding Array(Float32)')
  SETTINGS max_threads=32, max_insert_threads=8, max_insert_block_size=100000;
"
echo "LOAD_RC=$?"

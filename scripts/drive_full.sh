cd /home/ubuntu/embeddings
ls hn_in/shard_*.parquet | xargs -P 32 -n1 ./embed_one.sh
echo "ALL_SHARDS_PROCESSED"

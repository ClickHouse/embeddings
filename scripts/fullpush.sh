cd /home/ubuntu/embeddings
./clickhouse-new local --path . --config-file config.xml --allow_experimental_qbit_type 1 --query "
INSERT INTO FUNCTION remoteSecure('hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:9440','default.hackernews_embeddings_qwen3_8b','default','$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD')
SELECT m.update_time, m.id, m.deleted, m.type, m.by, m.time, m.text, m.dead, m.parent, m.poll, m.kids, m.url, m.score, m.title, m.parts, m.descendants, e.embedding
FROM emb_full_qwen e INNER JOIN hn_meta m ON e.id=m.id
SETTINGS max_threads=16, max_execution_time=0"
echo "FULLPUSH_RC=$?"

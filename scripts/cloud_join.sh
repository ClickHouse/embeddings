H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
./clickhouse-new client --host $H --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 --query "
INSERT INTO hackernews_embeddings_qwen3_8b
SELECT m.update_time, m.id, m.deleted, m.type, m.by, m.time, m.text, m.dead, m.parent, m.poll, m.kids, m.url, m.score, m.title, m.parts, m.descendants, e.embedding
FROM meta_stage m INNER JOIN emb_stage e ON m.id = e.id
SETTINGS join_algorithm='full_sorting_merge', max_execution_time=0"
echo "JOIN_RC=$?"

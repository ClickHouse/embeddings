cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
./clickhouse-new local --path . --config-file config.xml --allow_experimental_qbit_type 1 --query "
INSERT INTO FUNCTION remoteSecure('$H:9440','default.emb_stage','default','$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD')
SELECT id, embedding FROM emb_full_qwen
SETTINGS max_threads=32, max_insert_threads=16, max_execution_time=0"
echo "PUSHEMB_RC=$?"

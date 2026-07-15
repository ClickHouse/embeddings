cd /home/ubuntu/embeddings
./clickhouse-new local --path . --config-file config.xml --query "
INSERT INTO FUNCTION remoteSecure('hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:9440','default.meta_stage','default','$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD')
SELECT update_time, id, deleted, type, by, time, text, dead, parent, poll, kids, url, score, title, parts, descendants FROM hn_meta
SETTINGS max_threads=16"
echo "PUSHMETA_RC=$?"

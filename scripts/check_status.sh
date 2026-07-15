#!/bin/bash
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" --query "$1"; }
echo "meta_rows = $(CC 'SELECT count() FROM mmcommons.yfcc_metadata')"
echo "sqlite3 procs = $(pgrep -fc sqlite3)   meta-insert procs = $(pgrep -af 'mmcommons.yfcc_metadata' | grep -vc grep)"
echo "feature done-groups = $(wc -l < ingest.done 2>/dev/null)   driver alive = $(pgrep -fc ingest_features.sh)"
echo "--- feature row counts (loaded so far) ---"
CC "SELECT substr(name,1,22) AS tbl, formatReadableQuantity(total_rows) AS rows, formatReadableSize(total_bytes) AS size FROM system.tables WHERE database='mmcommons' AND total_rows>0 ORDER BY total_rows DESC LIMIT 12 FORMAT PrettyCompact"
echo "--- last driver events ---"; tail -3 ingest_features.log 2>/dev/null

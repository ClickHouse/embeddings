#!/bin/bash
idx="$1"; off="$2"; cd /home/ubuntu/embeddings
[ -f "ann_done/laion400m_$idx" ] && exit 0
BASE="https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings"
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
set -o pipefail
curl -s --retry 5 --retry-all-errors --max-time 1200 "$BASE/img_emb/img_emb_$idx.npy" \
 | python3 npyvec.py "$off" 2>/dev/null \
 | ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" \
     --query "INSERT INTO ann.laion400m_base SETTINGS insert_deduplication_token='laion400m_$idx' FORMAT RowBinary" 2>/dev/null
[ $? -eq 0 ] && touch "ann_done/laion400m_$idx" || echo "FAIL laion $idx"

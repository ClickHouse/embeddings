#!/bin/bash
table="$1"; url="$2"; typ="$3"; dim="$4"; rowbytes="$5"; lo="$6"; hi="$7"
cd /home/ubuntu/embeddings
[ -f "ann_done/${table}_${lo}" ] && exit 0
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
start=$((8 + lo*rowbytes)); end=$((8 + hi*rowbytes - 1))
set -o pipefail
curl -s --retry 5 --retry-all-errors -r ${start}-${end} "$url" \
 | python3 binvec.py "$typ" "$dim" "$lo" 2>/dev/null \
 | ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" \
     --query "INSERT INTO ann.${table} SETTINGS insert_deduplication_token='${table}_${lo}' FORMAT RowBinary" 2>/dev/null
[ $? -eq 0 ] && touch "ann_done/${table}_${lo}" || echo "FAIL ${table} ${lo}"

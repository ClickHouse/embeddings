#!/bin/bash
lo="$1"; hi="$2"; cd /home/ubuntu/embeddings
[ -f "thumb_done/$lo" ] && exit 0
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CHB=(./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD")
set -o pipefail
"${CHB[@]}" --query "SELECT md5 FROM mmcommons.image_blob WHERE md5>='$lo' AND md5<'$hi' FORMAT TSV" 2>/dev/null \
 | python3 thumbs_s3.py 64 2>/dev/null \
 | "${CHB[@]}" --query "INSERT INTO mmcommons.image_thumbs SETTINGS insert_deduplication_token='thumb_$lo' FORMAT Parquet" 2>/dev/null
[ $? -eq 0 ] && touch "thumb_done/$lo" || echo "FAIL $lo"

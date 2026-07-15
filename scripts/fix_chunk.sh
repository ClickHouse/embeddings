#!/bin/bash
cd /home/ubuntu/embeddings
lo="$1"; hi="$2"
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CH=(./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD")
set -o pipefail
f="/tmp/fix_${lo}.pq"
# 1) resize DISTINCT md5s to a local parquet FIRST (no table change yet)
"${CH[@]}" --query "SELECT DISTINCT md5 FROM mmcommons.image_blob WHERE md5>='$lo' AND md5<'$hi' FORMAT TSV" 2>/dev/null \
  | python3 thumbs_s3.py 96 2>/dev/null > "$f"
[ ! -s "$f" ] && { echo "EMPTY $lo"; exit 1; }
# 2) delete old rows in range, then insert the clean rebuild
"${CH[@]}" --query "DELETE FROM mmcommons.image_thumbs WHERE md5>='$lo' AND md5<'$hi'" 2>/dev/null || { echo "DELFAIL $lo"; exit 1; }
"${CH[@]}" --query "INSERT INTO mmcommons.image_thumbs SETTINGS insert_deduplication_token='fix2_${lo}' FORMAT Parquet" < "$f" 2>/dev/null \
  && { echo "OK $lo"; rm -f "$f"; } || echo "INSFAIL $lo"

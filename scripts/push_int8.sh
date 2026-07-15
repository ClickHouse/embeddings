#!/bin/bash
# Resumable parallel push of Int8 code shards to the Cloud table.
# Usage: push_int8.sh [parallelism] [glob]
set -u
P="${1:-12}"
GLOB="${2:-qcodes*/*.parquet}"
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CH=/home/ubuntu/embeddings/clickhouse-new
TBL=hackernews_embeddings_qwen3_8b_int8
DONE=/home/ubuntu/embeddings/pushed.log
touch "$DONE"

push_one() {
  local f="$1"
  grep -qxF "$f" "$DONE" 2>/dev/null && return 0
  if "$CH" local --query "
        INSERT INTO FUNCTION
          remoteSecure('$H:9440','default.$TBL','default','$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD')
          (id, q)
        SELECT id, CAST(q AS Array(Int8)) FROM file('$f', Parquet)
        SETTINGS max_insert_threads=2, max_threads=2
      " 2>>/tmp/push_int8.err; then
    echo "$f" >> "$DONE"
  else
    echo "FAIL $f" >&2
  fi
}
export -f push_one; export H CH TBL DONE CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD

cd /home/ubuntu/embeddings
ls $GLOB | xargs -P "$P" -I{} bash -c 'push_one "$@"' _ {}
echo "pushed_log_lines=$(wc -l < "$DONE")"

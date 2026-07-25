#!/usr/bin/env bash
# ngram skip-indexes so the search filter's substring match is fast (per-tile) instead of a full scan:
#   HackerNews : lowerUTF8(text) on default.hackernews           (full-text search over comments)
#   Photos     : lowerUTF8(arrayStringConcat(user_tags,' ')) on mmcommons.tags_by_md5  (tag search)
# ngrambf_v1(3,...) accelerates LIKE '%substr%' for substrings of length >= 3.
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --query "$1" 2>&1 | grep -vi "unknown setting"; }
echo "=== HN text ngram index $(date +%T) ==="
CL "ALTER TABLE default.hackernews ADD INDEX IF NOT EXISTS idx_text_ngram lowerUTF8(text) TYPE ngrambf_v1(3, 131072, 3, 0) GRANULARITY 1"
CL "ALTER TABLE default.hackernews MATERIALIZE INDEX idx_text_ngram SETTINGS mutations_sync=2"
echo "=== Photos tags ngram index $(date +%T) ==="
CL "ALTER TABLE mmcommons.tags_by_md5 ADD INDEX IF NOT EXISTS idx_tags_ngram lowerUTF8(arrayStringConcat(user_tags, ' ')) TYPE ngrambf_v1(3, 131072, 3, 0) GRANULARITY 1"
CL "ALTER TABLE mmcommons.tags_by_md5 MATERIALIZE INDEX idx_tags_ngram SETTINGS mutations_sync=2"
echo "=== SEARCH INDEXES DONE $(date +%T) ==="

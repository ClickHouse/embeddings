#!/usr/bin/env bash
# Corpus-wide token distribution for HackerNews. Used to score the most "contrasting" tokens of a top-K
# report (TF-IDF style: tokens frequent in the top-K but rare in the whole corpus). One row per distinct
# token, cnt = total occurrences across all HN item text. Tokeniser matches the app's per-top-K query:
#   arrayJoin(tokens(lowerUTF8(decodeHTMLComponent(extractTextFromHTML(text)))))
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --query "$1" 2>&1 | grep -vi "unknown setting"; }
echo "=== create table $(date +%T) ==="
CL "CREATE TABLE IF NOT EXISTS default.hackernews_tokens_stats (token String, cnt UInt64) ENGINE = SummingMergeTree ORDER BY token"
echo "=== fill (tokenise 43.6M items, group by token) $(date +%T) ==="
CL "INSERT INTO default.hackernews_tokens_stats
    SELECT arrayJoin(tokens(lowerUTF8(decodeHTMLComponent(extractTextFromHTML(text))))) AS token, count() AS cnt
    FROM default.hackernews
    WHERE text != ''
    GROUP BY token
    SETTINGS max_bytes_before_external_group_by = '20G', max_threads = 8"
echo "=== collapse SummingMergeTree $(date +%T) ==="
CL "OPTIMIZE TABLE default.hackernews_tokens_stats FINAL"
echo "=== verify $(date +%T) ==="
CL "SELECT count() distinct_tokens, sum(cnt) total_tokens, max(cnt) top FROM default.hackernews_tokens_stats"
CL "SELECT token, cnt FROM default.hackernews_tokens_stats ORDER BY cnt DESC LIMIT 8"
echo "=== HN TOKENS STATS DONE $(date +%T) ==="

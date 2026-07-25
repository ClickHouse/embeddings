#!/usr/bin/env bash
# Per-comment "top contrasting words" for the HackerNews "words" map mode (a word cloud analogue of Photos'
# "thumbs"). For each comment, the 5 highest-IDF tokens (rarity vs the whole corpus, cf>=100) — model-independent
# (depends only on comment text). Exposed as a dictionary keyed by id so the words-mode tile query can look up
# each record's words with dictGet (no per-projection column, no per-tile scan/join).
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --query "$1" 2>&1 | grep -vi "unknown setting"; }

echo "=== IDF dictionary (token -> idf, cf>=100)  $(date +%T) ==="
CL "CREATE DICTIONARY IF NOT EXISTS default.hackernews_idf_dict (token String, idf Float32)
    PRIMARY KEY token
    SOURCE(CLICKHOUSE(QUERY \$\$SELECT token, log(2355289038.0 / sum(cnt)) AS idf FROM default.hackernews_tokens_stats GROUP BY token HAVING sum(cnt) >= 100\$\$))
    LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"

echo "=== compute per-comment top-5 words  $(date +%T) ==="
CL "CREATE TABLE IF NOT EXISTS default.hackernews_topwords (id UInt32, words Array(String)) ENGINE = MergeTree ORDER BY id"
CL "TRUNCATE TABLE default.hackernews_topwords"
CL "INSERT INTO default.hackernews_topwords
    SELECT id,
      arraySlice(arrayMap(p -> p.1, arrayReverseSort(p -> p.2,
        arrayFilter(p -> p.2 > 0,
          arrayMap(t -> (t, dictGetFloat32('default.hackernews_idf_dict', 'idf', tuple(t))),
            arrayDistinct(tokens(lowerUTF8(decodeHTMLComponent(extractTextFromHTML(text))))))))), 1, 5) AS words
    FROM default.hackernews
    WHERE type = 'comment' AND text != ''
    SETTINGS max_execution_time = 0, max_threads = 16"

echo "=== topwords dictionary (id -> words)  $(date +%T) ==="
CL "CREATE DICTIONARY IF NOT EXISTS default.hackernews_topwords_dict (id UInt32, words Array(String))
    PRIMARY KEY id
    SOURCE(CLICKHOUSE(QUERY \$\$SELECT id, words FROM default.hackernews_topwords\$\$))
    LAYOUT(HASHED()) LIFETIME(0)"
CL "SYSTEM RELOAD DICTIONARY default.hackernews_topwords_dict"

echo "=== grant read access to the website users  $(date +%T) ==="
CL "GRANT SELECT ON default.hackernews_topwords TO website, website_thumbs"
CL "GRANT dictGet ON default.hackernews_topwords_dict TO website, website_thumbs"
CL "GRANT dictGet ON default.hackernews_idf_dict TO website, website_thumbs"

echo "=== verify  $(date +%T) ==="
CL "SELECT count() rows, round(avg(length(words)),2) avg_words FROM default.hackernews_topwords"
CL "SELECT dictGet('default.hackernews_topwords_dict', 'words', 4542385::UInt32)"
echo "=== HN TOPWORDS DONE  $(date +%T) ==="

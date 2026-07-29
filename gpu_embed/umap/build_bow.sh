#!/bin/bash
# Build hackernews_embeddings_bow via chunked server-side INSERT SELECT (hashing-trick bag-of-words,
# 8192-dim, L2-normalized, Array(BFloat16)). Chunked by id-range so each INSERT returns quickly
# (a single ~22min INSERT gets dropped by the HTTPS load balancer). Resumable + per-chunk dedup token.
set -u
H2=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
# NOTE: no insert_deduplication_token — a single per-chunk token dedups the chunk's later blocks and
# silently drops rows. Idempotency on retry is done by deleting the id-range before re-inserting.
ch(){ curl -sS --max-time 600 "https://${H2}:8443/" --user "default:${CH_PW:?set CH_PW}" --data-binary "$1"; }
q(){ curl -sS --max-time 120 "https://${H2}:8443/" --user "default:${CH_PW:?set CH_PW}" --data-binary "$1"; }

MAXID=$(q "SELECT max(id) FROM hackernews_embeddings_qwen3_8b")
W=${W:-1000000}
echo "max id=$MAXID width=$W  $(date -u +%T)"
lo=0
while [ "$lo" -le "$MAXID" ]; do
  hi=$((lo + W))
  # idempotent retry: clear this id-range before (re)inserting
  q "DELETE FROM hackernews_embeddings_bow WHERE id >= $lo AND id < $hi" >/dev/null 2>&1
  out=$(ch "INSERT INTO hackernews_embeddings_bow
    WITH 8192 AS vec_size,
      tokens(lowerUTF8(decodeHTMLComponent(extractTextFromHTML(text)))) AS toks,
      arrayMap(x -> cityHash64(x) % vec_size, toks) AS tokens_idx,
      arrayMap(j -> toFloat32(countEqual(tokens_idx, j)), range(vec_size)) AS vec,
      (vec / if(L2Norm(vec) > 0, L2Norm(vec), 1))::Array(BFloat16) AS embedding
    SELECT id, embedding FROM hackernews_embeddings_qwen3_8b
    WHERE id >= $lo AND id < $hi
    SETTINGS max_execution_time=0, max_block_size=2048, max_threads=12,
             min_insert_block_size_rows=20000, min_insert_block_size_bytes=0" 2>&1)
  if [ -n "$out" ]; then
    tries=$((${tries:-0}+1)); echo "chunk $lo-$hi ERROR (try $tries): $(echo "$out" | head -c 160)"
    [ "$tries" -ge 6 ] && { echo "chunk $lo-$hi GIVING UP after $tries"; lo=$hi; tries=0; continue; }
    sleep 15; continue
  fi
  tries=0
  n=$(q "SELECT count() FROM hackernews_embeddings_bow")
  echo "chunk $lo-$hi done; total rows=$n  $(date -u +%T)"
  lo=$hi
done
echo "BOW BUILD DONE $(date -u +%T); final rows=$(q "SELECT count() FROM hackernews_embeddings_bow")"

#!/bin/bash
# Chunked embed for one model. Usage: embed_chunked.sh <slug> <trunc> <chunk_rows> [batch_size]
cd /home/ubuntu/embeddings
slug="$1"; trunc="$2"; CHUNK="${3:-2000}"; BATCH="${4:-0}"; N=10000
BATCHOPT=""; [ "$BATCH" != "0" ] && BATCHOPT="--ai_function_embedding_max_batch_size $BATCH"
rm -f results/${slug}__c*.parquet
off=0; ci=0
while [ $off -lt $N ]; do
  ci=$((ci+1)); cf="results/${slug}__c$(printf '%02d' $ci).parquet"
  ok=0
  for attempt in $(seq 1 12); do
    ./clickhouse-new local --config-file or_config.xml --allow_experimental_ai_functions 1 \
      --ai_function_request_timeout_sec 120 --ai_function_max_retries 6 --ai_function_throw_on_error 1 $BATCHOPT \
      --query "
SELECT id, aiEmbed('or_${slug}', substringUTF8(doc,1,${trunc})) AS embedding
FROM (SELECT id, doc FROM file('hn_sample.parquet') ORDER BY id LIMIT ${CHUNK} OFFSET ${off})
INTO OUTFILE '${cf}' TRUNCATE FORMAT Parquet" >/dev/null 2>"logs/${slug}_c${ci}.log"
    rc=$?
    have=0; [ -f "$cf" ] && have=$(./clickhouse-new local --query "SELECT count() FROM file('$cf')" 2>/dev/null)
    if [ $rc -eq 0 ] && [ "$have" -gt 0 ]; then echo "  chunk $ci off=$off rows=$have ok(attempt=$attempt)"; ok=1; break; fi
    rm -f "$cf"; sleep 8
  done
  [ $ok -eq 1 ] || { echo "CHUNK_FAIL $slug chunk $ci off=$off"; exit 1; }
  off=$((off+CHUNK))
done
./clickhouse-new local --query "SELECT id, embedding FROM file('results/${slug}__c*.parquet') ORDER BY id INTO OUTFILE 'results/${slug}.parquet' TRUNCATE FORMAT Parquet" 2>&1 | tail -1
final=$(./clickhouse-new local --query "SELECT count() FROM file('results/${slug}.parquet')" 2>/dev/null)
echo "MERGED $slug rows=$final"
[ "$final" = "$N" ] && rm -f results/${slug}__c*.parquet

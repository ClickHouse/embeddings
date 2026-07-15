#!/bin/bash
shard="$1"; cd /home/ubuntu/embeddings; CH=./clickhouse-new
base=$(basename "$shard"); whole="results/qwen_full/$base"
inc=$($CH local --query "SELECT count() FROM file('$shard')" 2>/dev/null)
# legacy whole-shard output already complete -> skip
if [ -f "$whole" ]; then wc=$($CH local --query "SELECT count() FROM file('$whole')" 2>/dev/null); [ "$wc" = "$inc" ] && { echo "SKIP $base ($wc)"; exit 0; }; fi
CHUNK=8000; off=0; allok=1
while [ "$off" -lt "$inc" ]; do
  cf="results/qwen_full/${base%.parquet}_c$(printf '%07d' $off).parquet"
  exp=$CHUNK; [ $((inc-off)) -lt $CHUNK ] && exp=$((inc-off))
  oc=0; [ -f "$cf" ] && oc=$($CH local --query "SELECT count() FROM file('$cf')" 2>/dev/null)
  if [ "$oc" = "$exp" ]; then off=$((off+CHUNK)); continue; fi
  ok=0
  for a in $(seq 1 12); do
    $CH local --config-file ci_config.xml --allow_experimental_ai_functions 1 \
      --ai_function_request_timeout_sec 300 --ai_function_max_retries 8 --ai_function_retry_initial_delay_ms 2000 \
      --ai_function_embedding_max_batch_size 1024 --ai_function_throw_on_error 1 \
      --ai_function_max_input_tokens_per_query 100000000 --ai_function_max_output_tokens_per_query 100000000 \
      --query "SELECT id, aiEmbed('ci_qwen3_embedding_8b', substringUTF8(doc,1,30000)) AS embedding FROM (SELECT id,doc FROM file('$shard') ORDER BY id LIMIT $CHUNK OFFSET $off) INTO OUTFILE '$cf' TRUNCATE FORMAT Parquet" >"logs/qf_${base}_$off.log" 2>&1
    rc=$?; oc=0; [ -f "$cf" ] && oc=$($CH local --query "SELECT count() FROM file('$cf')" 2>/dev/null)
    [ $rc -eq 0 ] && [ "$oc" = "$exp" ] && { ok=1; break; }
    rm -f "$cf"; sleep 6
  done
  [ $ok -eq 1 ] || { echo "FAIL $base off=$off a=$a"; allok=0; break; }
  off=$((off+CHUNK))
done
[ $allok -eq 1 ] && echo "DONE $base ($inc subchunked)"

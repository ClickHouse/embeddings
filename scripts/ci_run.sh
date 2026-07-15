#!/bin/bash
cd /home/ubuntu/embeddings
CH=./clickhouse-new; N=10000; ATTEMPTS=4
run_one(){
  local slug="$1" trunc="$2" out="results/ci_${slug}.parquet" log="logs/ci_${slug}.log"
  if [ -f "$out" ]; then local h=$($CH local --query "SELECT count() FROM file('$out')" 2>/dev/null); [ "$h" = "$N" ] && { echo "SKIP ci_$slug"; return; }; fi
  for a in $(seq 1 $ATTEMPTS); do
    local s=$(date +%s)
    $CH local --config-file ci_config.xml --allow_experimental_ai_functions 1 \
      --ai_function_request_timeout_sec 120 --ai_function_max_retries 6 --ai_function_retry_initial_delay_ms 2000 --ai_function_throw_on_error 1 \
      --query "SELECT id, aiEmbed('ci_${slug}', substringUTF8(doc,1,${trunc})) AS embedding FROM file('hn_sample.parquet') INTO OUTFILE '${out}' TRUNCATE FORMAT Parquet" >"$log" 2>&1
    local rc=$? h=0; [ -f "$out" ] && h=$($CH local --query "SELECT count() FROM file('$out')" 2>/dev/null)
    if [ $rc -eq 0 ] && [ "$h" = "$N" ]; then echo "DONE ci_$slug rows=$h $(( $(date +%s)-s ))s attempt=$a"; return; fi
    echo "RETRY ci_$slug a=$a rc=$rc rows=$h :: $(tail -1 "$log"|cut -c1-80)"; rm -f "$out"; sleep 6
  done
  echo "FAIL ci_$slug"
}
export -f run_one; export CH N ATTEMPTS
while IFS=$'\t' read -r slug trunc model; do run_one "$slug" "$trunc" & sleep 2; done < ci_jobs.tsv
wait; echo ALL_DONE

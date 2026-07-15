#!/bin/bash
# Parallel embedding driver with per-model retry. model -> results/<slug>.parquet (id, embedding).
cd /home/ubuntu/embeddings
CH=./clickhouse-new
MAXJOBS=5
N=10000
ATTEMPTS=4

run_one() {
  local slug="$1" trunc="$2" model="$3"
  local out="results/${slug}.parquet" log="logs/${slug}.log"
  if [ -f "$out" ]; then
    local have=$($CH local --query "SELECT count() FROM file('$out')" 2>/dev/null)
    [ "$have" = "$N" ] && { echo "SKIP $slug (have $have)"; return 0; }
  fi
  local attempt=1
  while [ $attempt -le $ATTEMPTS ]; do
    local start=$(date +%s)
    $CH local --config-file or_config.xml --allow_experimental_ai_functions 1 \
        --ai_function_request_timeout_sec 120 --ai_function_max_retries 6 \
        --ai_function_retry_initial_delay_ms 2000 --ai_function_throw_on_error 1 \
        --query "
SELECT id, aiEmbed('or_${slug}', substringUTF8(doc, 1, ${trunc})) AS embedding
FROM file('hn_sample.parquet')
INTO OUTFILE '${out}' TRUNCATE FORMAT Parquet
" >"$log" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - start ))
    local have=0; [ -f "$out" ] && have=$($CH local --query "SELECT count() FROM file('$out')" 2>/dev/null)
    if [ $rc -eq 0 ] && [ "$have" = "$N" ]; then
      echo "DONE $slug rows=$have ${dur}s attempt=$attempt"; return 0
    fi
    echo "RETRY $slug attempt=$attempt rc=$rc rows=$have ${dur}s :: $(tail -1 "$log" | cut -c1-90)"
    rm -f "$out"; sleep 6; attempt=$((attempt+1))
  done
  echo "FAIL $slug after $ATTEMPTS attempts"
}
export -f run_one; export CH N ATTEMPTS

while IFS=$'\t' read -r slug trunc model; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  run_one "$slug" "$trunc" "$model" &
  sleep 2   # stagger launches to avoid startup burst
done < jobs.tsv
wait
echo "ALL_WORKERS_FINISHED"

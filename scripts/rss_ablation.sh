cd /home/ubuntu/embeddings
run_scan() {
  T="$1"; PB="$2"
  ./clickhouse-new local --path . --allow_experimental_qbit_type 1 --query "
    WITH (SELECT q::Array(BFloat16) FROM file('qvec.parquet')) AS qv
    SELECT id, cosineDistanceTransposed(embedding, qv, 16) AS d FROM emb_full_qwen ORDER BY d ASC LIMIT 5
    SETTINGS max_threads=$T, preferred_block_size_bytes=$PB FORMAT Null" >/dev/null 2>&1 &
  qpid=$!
  max=0
  for i in $(seq 1 75); do
    kill -0 $qpid 2>/dev/null || break
    rss=$(ps --no-headers -o rss= --ppid $qpid 2>/dev/null | awk '{s+=$1} END{print s}')
    self=$(ps --no-headers -o rss= -p $qpid 2>/dev/null | tr -d ' ')
    tot=$(( ${rss:-0} + ${self:-0} ))
    [ "$tot" -gt "$max" ] && max=$tot
    sleep 2
  done
  kill -9 $qpid 2>/dev/null; wait $qpid 2>/dev/null
  echo "max_threads=$T preferred_block_size_bytes=$PB  -> peak_RSS=$(numfmt --to=iec $((max*1024)))"
}
echo "=== peak RSS vs max_threads (full scan, bits=16) ==="
run_scan 64 1000000
run_scan 32 1000000
run_scan 8  1000000
echo "=== effect of much smaller block size (threads=64) ==="
run_scan 64 65536
echo "ABLATION_DONE"

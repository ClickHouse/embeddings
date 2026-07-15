cd /home/ubuntu/embeddings
./clickhouse-new local --path . --allow_experimental_qbit_type 1 --query "
SET allow_introspection_functions=1;
SELECT id, cosineDistanceTransposed(embedding, (SELECT q::Array(BFloat16) FROM file('qvec.parquet')), 16) AS d
FROM emb_full_qwen ORDER BY d ASC LIMIT 5
SETTINGS max_threads=64, memory_profiler_sample_probability=1.0, memory_profiler_sample_min_allocation_size=1048576
FORMAT Null;
SYSTEM FLUSH LOGS;
SELECT formatReadableSize(sum(size)) AS resident, count() AS allocs,
       arrayStringConcat(arrayMap(a -> demangle(addressToSymbol(a)), arraySlice(trace,1,7)), ' <- ') AS stack
FROM system.trace_log WHERE trace_type='MemorySample' AND size > 0
GROUP BY trace ORDER BY sum(size) DESC LIMIT 18;
"
echo "PROFILE_RC=$?"

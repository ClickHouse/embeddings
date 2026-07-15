#!/bin/bash
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CC() { ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 "$@"; }
cd /home/ubuntu/embeddings

echo "########## SEARCH, PARALLEL REPLICAS (automatic mode OFF), bits=16 ##########"
CC --query_id pr_fix --time --query "
  SELECT id, by,
         round(cosineDistanceTransposed(embedding,(SELECT q FROM q_clickhouse LIMIT 1),16),4) AS dist,
         substring(text,1,55) AS t
  FROM hackernews_embeddings_qwen3_8b
  WHERE text NOT ILIKE '%clickhouse%'
  ORDER BY dist ASC LIMIT 5
  SETTINGS enable_parallel_replicas=1, automatic_parallel_replicas_mode=0,
           max_parallel_replicas=3, cluster_for_parallel_replicas='default'"

echo
echo "########## engagement: rows read per replica ##########"
CC --query "SYSTEM FLUSH LOGS ON CLUSTER 'default'" >/dev/null 2>&1
CC --query "
  SELECT hostName() AS host, is_initial_query AS init, query_duration_ms AS ms, read_rows,
    ProfileEvents['ParallelReplicasUsedCount'] AS pr_used,
    ProfileEvents['ParallelReplicasAvailableCount'] AS pr_avail
  FROM clusterAllReplicas('default', system.query_log)
  WHERE query_id='pr_fix' AND type='QueryFinish'
  ORDER BY init DESC, host FORMAT PrettyCompact"
echo "DONE"

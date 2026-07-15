#!/bin/bash
# Sweep cosineDistanceTransposed bits on Cloud to find the speed/accuracy tradeoff.
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CC() { ./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --allow_experimental_qbit_type 1 "$@"; }
cd /home/ubuntu/embeddings
for B in 16 8 4 2 1; do
  echo "########## bits=$B ##########"
  CC --time --query "
    SELECT id, by,
           round(cosineDistanceTransposed(embedding,(SELECT q FROM q_clickhouse LIMIT 1),$B),4) AS dist,
           substring(text,1,55) AS t
    FROM hackernews_embeddings_qwen3_8b
    WHERE text NOT ILIKE '%clickhouse%'
    ORDER BY dist ASC LIMIT 5"
  echo
done
echo "DONE"

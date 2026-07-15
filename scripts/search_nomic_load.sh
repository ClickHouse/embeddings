#!/bin/bash
# Populate mmcommons.search_nomic = yfcc_metadata columns + nomic embedding (QBit),
# INNER JOIN on the stripped md5, only for photos that have a nomic embedding.
# 16 md5-prefix buckets: memory-safe (per-bucket build ~9 GiB) and idempotent (dedup token).
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" --allow_experimental_qbit_type 1 "$@" 2>&1 | grep -v "Unknown settings"; }
LO="0123456789abcdef"; HI="123456789abcdefg"
t0=$(date +%s)
for i in $(seq 0 15); do
  lo="${LO:$i:1}"; hi="${HI:$i:1}"; bs=$(date +%s)
  echo "=== bucket '$lo' start $(date -u +%T) ==="
  CC --query "
INSERT INTO mmcommons.search_nomic
  (photo_id,user_nsid,user_nickname,date_taken,date_uploaded,capture_device,title,description,user_tags,machine_tags,lon,lat,geo_accuracy,page_url,download_url,license_name,license_url,server_id,farm_id,secret,secret_original,ext,media, embedding)
SELECT m.photo_id,m.user_nsid,m.user_nickname,m.date_taken,m.date_uploaded,m.capture_device,m.title,m.description,m.user_tags,m.machine_tags,m.lon,m.lat,m.geo_accuracy,m.page_url,m.download_url,m.license_name,m.license_url,m.server_id,m.farm_id,m.secret,m.secret_original,m.ext,m.media,
       CAST(e.embedding AS QBit(BFloat16,768,16))
FROM mmcommons.yfcc_metadata AS m
INNER JOIN (
  SELECT md5, any(embedding) AS embedding
  FROM mmcommons.emb_nomic
  WHERE md5 >= '$lo' AND md5 < '$hi'
  GROUP BY md5
) AS e
ON e.md5 = arrayStringConcat(arrayMap(k -> if(substring(lower(hex(MD5(m.download_url))) AS hx,(k*2)-1,1)='0', substring(hx,k*2,1), substring(hx,(k*2)-1,2)), range(1,17)))
SETTINGS optimize_aggregation_in_order=1, max_execution_time=0, max_insert_threads=4, insert_deduplication_token='search_nomic_$lo'"
  rc=$?
  n=$(CC --query "SELECT count() FROM mmcommons.search_nomic")
  echo "=== bucket '$lo' rc=$rc  $(( $(date +%s)-bs ))s  total_rows=$n  elapsed=$(( $(date +%s)-t0 ))s ==="
done
echo "ALL DONE. final rows: $(CC --query 'SELECT count() FROM mmcommons.search_nomic')  total $(( $(date +%s)-t0 ))s"

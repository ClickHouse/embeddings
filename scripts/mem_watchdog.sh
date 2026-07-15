#!/bin/bash
# Monitors max per-replica memory; auto-throttles the image driver to protect the 3x120GiB service.
# Thresholds (max replica CGroupMemoryUsed): >=100 -> N=24, >=108 -> N=16, >=115 -> pause; resume@16 when <85.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
maxmem(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" --query \
  "SELECT toString(round(max(value)/1073741824,1)) FROM clusterAllReplicas(default, system.asynchronous_metrics) WHERE metric='CGroupMemoryUsed'" 2>/dev/null; }
relaunch(){ # $1=N $2=mem
  pkill -9 -f img_worker.sh 2>/dev/null; pkill -9 -f 'ingest_images.sh' 2>/dev/null
  pkill -9 -f 'INSERT INTO mmcommons.image_blob' 2>/dev/null; sleep 3
  nohup bash ingest_images.sh "$1" >> image_load.log 2>&1 &
  echo "$(date -u +%H:%M:%S) ACTION -> N=$1 (mem ${2}GiB, done=$(wc -l < image.done))" >> mem_watch.log
}
cur=32
echo "$(date -u +%H:%M:%S) watchdog start N=$cur" >> mem_watch.log
for i in $(seq 1 2880); do
  [ "$(wc -l < image.done 2>/dev/null)" -ge 4096 ] && { echo "$(date -u +%H:%M:%S) IMAGE LOAD COMPLETE" >> mem_watch.log; break; }
  m=$(maxmem); mi=${m%.*}
  [ -z "$mi" ] && { sleep 60; continue; }
  echo "$(date -u +%H:%M:%S) max_mem=${m}GiB N=$cur done=$(wc -l < image.done)" >> mem_watch.log
  if   [ "$mi" -ge 115 ] && [ "$cur" -ne 0 ]; then
        pkill -9 -f img_worker.sh; pkill -9 -f 'ingest_images.sh'; pkill -9 -f 'INSERT INTO mmcommons.image_blob'; cur=0
        echo "$(date -u +%H:%M:%S) CRITICAL PAUSE at ${m}GiB" >> mem_watch.log; sleep 120; continue
  elif [ "$cur" -eq 0 ] && [ "$mi" -lt 85 ]; then relaunch 16 "$m"; cur=16; sleep 150; continue
  elif [ "$mi" -ge 108 ] && [ "$cur" -gt 16 ]; then relaunch 16 "$m"; cur=16; sleep 150; continue
  elif [ "$mi" -ge 100 ] && [ "$cur" -gt 24 ]; then relaunch 24 "$m"; cur=24; sleep 150; continue
  fi
  sleep 60
done
echo "$(date -u +%H:%M:%S) watchdog exit" >> mem_watch.log

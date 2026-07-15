#!/bin/bash
cd /home/ubuntu/embeddings
for pass in 1 2 3 4 5 6; do
  d=$(ls ann_done 2>/dev/null | grep -c laion400m_)
  echo "--- pass $pass: $d/410 shards done ---"
  [ "$d" -ge 410 ] && { echo "ALL LAION SHARDS DONE"; break; }
  bash laionrun.sh 16 >/dev/null 2>&1
done
echo "LAION MOPUP DONE: $(ls ann_done|grep -c laion400m_)/410"

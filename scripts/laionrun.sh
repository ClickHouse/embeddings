#!/bin/bash
cd /home/ubuntu/embeddings
W="${1:-16}"
awk '{print $1, $2}' /tmp/laion_shards.txt > /tmp/laion_pairs.txt
xargs -P "$W" -a /tmp/laion_pairs.txt -n2 bash laionchunk.sh
echo "LAION RUN DONE: done=$(ls ann_done|grep -c laion400m_)"

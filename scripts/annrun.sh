#!/bin/bash
# args: table url typ dim rowbytes N chunkrows W
cd /home/ubuntu/embeddings
table="$1"; url="$2"; typ="$3"; dim="$4"; rowbytes="$5"; N="$6"; C="$7"; W="${8:-16}"
python3 -c "
N=$N; C=$C
lo=0
while lo<N:
    hi=min(lo+C,N); print('$table','$url','$typ','$dim','$rowbytes',lo,hi); lo=hi
" > /tmp/ann_${table}_chunks.txt
echo "$(wc -l < /tmp/ann_${table}_chunks.txt) chunks, W=$W"
xargs -P "$W" -a /tmp/ann_${table}_chunks.txt -n7 bash annchunk.sh
echo "RUN DONE ${table}: done=$(ls ann_done|grep -c "^${table}_")"

#!/bin/bash
cd /home/ubuntu/embeddings
W=${1:-48}
mkdir -p thumb_done
python3 -c "
for i in range(4096): print('%03x'%i, ('%03x'%(i+1)) if i<4095 else 'g')
" > thumb_chunks.txt
echo "launch W=$W over 4096 chunks (S3 source); done: $(ls thumb_done|wc -l)"
xargs -P "$W" -a thumb_chunks.txt -n2 bash thumb_s3_chunk.sh
echo "ALL DONE"

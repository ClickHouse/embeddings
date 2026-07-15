#!/bin/bash
cd /home/ubuntu/embeddings
mkdir -p thumb2_done
W="${1:-80}"
python3 -c "
for i in range(4096): print('%03x'%i, ('%03x'%(i+1)) if i<4095 else 'g')
" > thumb2_chunks.txt
xargs -P "$W" -a thumb2_chunks.txt -n2 bash thumb_s3_v2_chunk.sh
echo "run finished; done markers: $(ls thumb2_done|wc -l)/4096"

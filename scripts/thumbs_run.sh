#!/bin/bash
cd /home/ubuntu/embeddings
W=${1:-16}
mkdir -p thumb_done
python3 -c "
for i in range(4096):
    print('%03x'%i, ('%03x'%(i+1)) if i<4095 else 'g')
" > thumb_chunks.txt
echo "launch W=$W over 4096 md5 chunks; already done: $(ls thumb_done 2>/dev/null|wc -l)"
xargs -P "$W" -a thumb_chunks.txt -n2 bash thumb_chunk.sh
echo "ALL CHUNKS DONE"

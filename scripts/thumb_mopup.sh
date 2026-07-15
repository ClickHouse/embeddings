#!/bin/bash
cd /home/ubuntu/embeddings
for pass in 1 2 3 4 5 6; do
  python3 -c "
import os
done=set(os.listdir('thumb_done'))
miss=[('%03x'%i) for i in range(4096) if ('%03x'%i) not in done]
lines=[f\"{m} {'%03x'%(int(m,16)+1) if int(m,16)<4095 else 'g'}\" for m in miss]
open('/tmp/mop.txt','w').write('\n'.join(lines)+('\n' if lines else ''))
print(f'--- pass $pass: {len(miss)} chunks remaining ---', flush=True)
"
  n=$(grep -c . /tmp/mop.txt 2>/dev/null || echo 0)
  [ "$n" -eq 0 ] && { echo "ALL CHUNKS DONE"; break; }
  xargs -P 64 -a /tmp/mop.txt -n2 bash thumb_s3_chunk.sh
done
echo "mopup finished; done markers: $(ls thumb_done|wc -l)/4096"

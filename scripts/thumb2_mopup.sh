#!/bin/bash
cd /home/ubuntu/embeddings
for pass in 1 2 3 4 5 6; do
  python3 -c "
import os
done=set(os.listdir('thumb2_done'))
miss=[('%03x'%i) for i in range(4096) if ('%03x'%i) not in done]
lines=[f\"{m} {'%03x'%(int(m,16)+1) if int(m,16)<4095 else 'g'}\" for m in miss]
open('/tmp/mop2.txt','w').write('\n'.join(lines)+('\n' if lines else ''))
print(f'--- pass $pass: {len(miss)} chunks remaining ---', flush=True)
"
  n=$(grep -c . /tmp/mop2.txt 2>/dev/null || echo 0)
  [ "$n" = "0" ] && { echo "ALL DONE"; break; }
  xargs -P 40 -a /tmp/mop2.txt -n2 bash thumb_s3_v2_chunk.sh
done
echo "mopup finished; done markers: $(ls thumb2_done|wc -l)/4096"

#!/bin/bash
# Parallel, resumable image-blob loader: 4096 three-hex prefixes -> xargs -P N concurrent INSERTs.
cd /home/ubuntu/embeddings
N=${1:-16}
touch image.done
python3 -c "print('\n'.join('%03x'%i for i in range(4096)))" > image_prefixes.txt
echo "START images N=$N done_so_far=$(wc -l < image.done) $(date -u +%H:%M:%S)" >> image_load.log
xargs -P "$N" -I{} bash img_worker.sh {} < image_prefixes.txt
echo "IMAGE_LOAD_COMPLETE done=$(wc -l < image.done)/4096 $(date -u +%H:%M:%S)" >> image_load.log

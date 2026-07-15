#!/bin/bash
cd /home/ubuntu/embeddings
YB=https://storage.yandexcloud.net/yandex-research/ann-datasets
AZ=https://comp21storage.z5.web.core.windows.net/comp21
run(){ echo "=== $(date +%T) START $1 ==="; bash annrun.sh "$@"; echo "=== $(date +%T) END $1 ==="; }
run deep1b_base       "$YB/DEEP/base.1B.fbin"                 f32 96  384 1000000000 4000000 24
run msturing1b_base   "$AZ/MSFT-TURING-ANNS/base1b.fbin"      f32 100 400 1000000000 4000000 24
run msspacev1b_base   "$AZ/spacev1b/spacev1b_base.i8bin"      i8  100 100 1000000000 10000000 24
run text2image1b_base "$YB/T2I/base.1B.fbin"                  f32 200 800 1000000000 2000000 24
echo "ALL BASE DONE"

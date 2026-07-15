#!/bin/bash
lo="$1"; hi="$2"; cd /home/ubuntu/embeddings
[ -f "thumb2_done/$lo" ] && exit 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1  # 1 core/proc; parallelism from W procs
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CHB=(./clickhouse-new client --host "$H" --secure --user default --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD")
set -o pipefail
"${CHB[@]}" --query "SELECT DISTINCT md5 FROM mmcommons.image_blob WHERE md5>='$lo' AND md5<'$hi' FORMAT TSV" 2>/dev/null \
 | python3 thumbs_s3_v2.py 64 2>/dev/null \
 | "${CHB[@]}" --query "INSERT INTO mmcommons.image_thumbs2 SETTINGS insert_deduplication_token='thumb2_$lo' FORMAT Parquet" 2>/dev/null
[ $? -eq 0 ] && touch "thumb2_done/$lo" || echo "FAIL $lo"

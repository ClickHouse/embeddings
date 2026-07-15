#!/bin/bash
# Fix feat_sift prefix 'e': delete partial+dup rows, reload all e** shards EXCEPT corrupt ea6.sift.gz.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
S3=https://multimedia-commons.s3.us-west-2.amazonaws.com
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" "$@"; }
ins(){ CC --max_insert_threads 4 --max_threads 8 --max_memory_usage 20000000000 --query \
  "INSERT INTO mmcommons.feat_sift SELECT arr[1], arrayMap(x->toFloat32OrZero(x), arraySlice(arr,2)) FROM (SELECT splitByChar(',', line) AS arr FROM s3('$S3/features/image/sift/$1', NOSIGN,'LineAsString','line String'))"; }
echo "START $(date -u +%H:%M:%S) delete existing 'e' rows" >> sift_e_fix.log
CC --mutations_sync 1 --query "ALTER TABLE mmcommons.feat_sift DELETE WHERE startsWith(md5,'e')"
echo "DELETE rc=$? $(date -u +%H:%M:%S) e_rows_after_delete=$(CC --query "SELECT count() FROM mmcommons.feat_sift WHERE startsWith(md5,'e')")" >> sift_e_fix.log
for g in e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 eb ec ed ee ef; do
  ins "${g}?.sift.gz"; echo "INSERT ${g}? rc=$? $(date -u +%H:%M:%S)" >> sift_e_fix.log
done
ins "ea{0,1,2,3,4,5,7,8,9,a,b,c,d,e,f}.sift.gz"; echo "INSERT ea-minus-ea6 rc=$? $(date -u +%H:%M:%S)" >> sift_e_fix.log
echo "DONE $(date -u +%H:%M:%S) e_rows=$(CC --query "SELECT count() FROM mmcommons.feat_sift WHERE startsWith(md5,'e')") e_distinct=$(CC --query "SELECT uniqExact(md5) FROM mmcommons.feat_sift WHERE startsWith(md5,'e')")" >> sift_e_fix.log

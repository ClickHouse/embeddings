#!/bin/bash
# Resumable server-side ingest of Multimedia Commons features from public S3 into ClickHouse Cloud.
# Each (feature, hex-group) is one INSERT...SELECT FROM s3(); progress logged to ingest.done.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
S3=https://multimedia-commons.s3.us-west-2.amazonaws.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
DONE=ingest.done; touch $DONE
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" \
      --max_execution_time 0 --max_insert_threads 4 --max_threads 8 \
      --input_format_parallel_parsing 1 --query "$1"; }

# spec: kind|table|s3prefix|ext
SPECS=(
"cnn|feat_hybrid_cnn|features/image/hybrid-cnn|txt.gz"
"lire|feat_img_bf|features/image/bf|bf"
"lire|feat_img_tamura|features/image/tamura|tamura"
"lire|feat_img_scalablecolor|features/image/scalablecolor|scalablecolor"
"lire|feat_img_edgehistogram|features/image/edgehistogram|edgehistogram"
"lire|feat_img_col|features/image/col|col"
"lire|feat_img_cedd|features/image/cedd|cedd"
"lire|feat_img_fcth|features/image/fcth|fcth"
"lire|feat_img_gabor|features/image/gabor|gabor"
"lire|feat_img_RGB|features/image/RGB|RGB"
"lire|feat_img_ophist|features/image/ophist|ophist"
"lire|feat_img_jhist|features/image/jhist|jhist"
"lire|feat_img_jophist|features/image/jophist|jophist"
"lire|feat_img_gist|features/image/gist|gist"
"lire|feat_img_acc|features/image/acc|acc"
"kf|feat_kf_bf|features/keyframe/bf|bf"
"kf|feat_kf_tamura|features/keyframe/tamura|tamura"
"kf|feat_kf_scalablecolor|features/keyframe/scalablecolor|scalablecolor"
"kf|feat_kf_edgehistogram|features/keyframe/edgehistogram|edgehistogram"
"kf|feat_kf_col|features/keyframe/col|col"
"kf|feat_kf_cedd|features/keyframe/cedd|cedd"
"kf|feat_kf_fcth|features/keyframe/fcth|fcth"
"kf|feat_kf_gabor|features/keyframe/gabor|gabor"
"kf|feat_kf_RGB|features/keyframe/RGB|RGB"
"kf|feat_kf_ophist|features/keyframe/ophist|ophist"
"kf|feat_kf_jhist|features/keyframe/jhist|jhist"
"kf|feat_kf_jophist|features/keyframe/jophist|jophist"
"kf|feat_kf_acc|features/keyframe/acc|acc"
"sift|feat_sift|features/image/sift|sift.gz"
)

build_sql(){ # kind table glob
  local kind="$1" table="$2" glob="$3"
  case "$kind" in
   cnn)  echo "INSERT INTO mmcommons.$table SELECT toUInt64OrZero(arr[1]), arr[2], arrayMap(x->toFloat32OrZero(x), arraySlice(arr,3)) FROM (SELECT splitByChar('\t', line) AS arr FROM s3('$glob', NOSIGN,'LineAsString','line String'))";;
   sift) echo "INSERT INTO mmcommons.$table SELECT arr[1], arrayMap(x->toFloat32OrZero(x), arraySlice(arr,2)) FROM (SELECT splitByChar(',', line) AS arr FROM s3('$glob', NOSIGN,'LineAsString','line String'))";;
   lire) echo "INSERT INTO mmcommons.$table SELECT arr[1], arrayMap(x->toFloat32OrZero(x), arraySlice(arr,4)) FROM (SELECT splitByChar(' ', replaceAll(line,'\t',' ')) AS arr FROM s3('$glob', NOSIGN,'LineAsString','line String'))";;
   kf)   echo "INSERT INTO mmcommons.$table SELECT arr[1], arrayMap(x->toFloat32OrZero(x), arraySlice(arr,4)) FROM (SELECT splitByChar(' ', replaceAll(line,'\t',' ')) AS arr FROM s3('$glob', NOSIGN,'LineAsString','line String'))";;
  esac
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r kind table s3prefix ext <<< "$spec"
  for g in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
    tag="$table $g"
    grep -qxF "$tag" $DONE && continue
    glob="$S3/$s3prefix/${g}*.${ext}"
    ok=0
    for attempt in 1 2; do
      if CC "$(build_sql "$kind" "$table" "$glob")" 2>>ingest_err.log; then ok=1; break; fi
      echo "RETRY $tag attempt=$attempt $(date -u +%H:%M:%S)" >> ingest_features.log; sleep 10
    done
    if [ $ok -eq 1 ]; then echo "$tag" >> $DONE; echo "DONE $tag $(date -u +%H:%M:%S)" >> ingest_features.log
    else echo "FAIL $tag $(date -u +%H:%M:%S)" >> ingest_features.log; fi
  done
  echo "FEATURE_COMPLETE $table $(date -u +%H:%M:%S)" >> ingest_features.log
done
echo "ALL_FEATURES_DONE $(date -u +%H:%M:%S)" >> ingest_features.log

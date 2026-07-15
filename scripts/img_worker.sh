#!/bin/bash
# Load one 3-hex prefix of images into mmcommons.image_blob via s3() RawBLOB wildcard.
p="$1"
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
S3=https://multimedia-commons.s3.us-west-2.amazonaws.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
grep -qxF "$p" image.done 2>/dev/null && exit 0
out=$(./clickhouse-new client --host "$H" --secure --user default --password "$PW" \
  --max_insert_threads 4 --max_threads 6 --s3_max_connections 24 --max_execution_time 0 \
  --query "INSERT INTO mmcommons.image_blob
           SELECT splitByChar('.', _file)[1] AS md5, content
           FROM s3('$S3/data/images/$p/*/*.jpg', NOSIGN, 'RawBLOB', 'content String')" 2>&1)
if [ $? -eq 0 ]; then echo "$p" >> image.done
else echo "FAIL $p :: $(echo "$out" | tail -1 | cut -c1-120)" >> image_err.log; fi

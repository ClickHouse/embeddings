#!/bin/bash
# Push the per-prefix Parquet embeddings into ClickHouse Cloud (mmcommons.emb_*).
# Requires: ./clickhouse binary (from setup.sh), env CH_HOST + CH_PW set.
# Usage: CH_HOST=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com CH_PW=... ./load_to_clickhouse.sh [models]
set -e
OUT=${OUT:-/mnt/emb}
MODELS=${1:-siglip2 clip nomic nemotron}; MODELS=${MODELS//,/ }
: "${CH_HOST:?set CH_HOST}"; : "${CH_PW:?set CH_PW}"
CC(){ ./clickhouse client --host "$CH_HOST" --secure --user default --password "$CH_PW" "$@"; }
# create tables
CC --queries-file schema.sql
touch loaded.done
one(){ # $1=model $2=parquet
  grep -qxF "$1/$2" loaded.done && return 0
  ./clickhouse client --host "$CH_HOST" --secure --user default --password "$CH_PW" \
     --query "INSERT INTO mmcommons.emb_$1 SELECT md5, CAST(embedding AS Array(BFloat16)) FROM input('md5 String, embedding Array(Float32)') FORMAT Parquet" \
     < "$OUT/$1/$2" && echo "$1/$2" >> loaded.done
}
export -f one; export OUT CH_HOST CH_PW
for m in $MODELS; do
  echo "loading emb_$m ..."
  ls "$OUT/$m" 2>/dev/null | grep '\.parquet$' | xargs -P8 -I{} bash -c 'one "$0" "$1"' "$m" {}
  echo "emb_$m rows: $(CC --query "SELECT count() FROM mmcommons.emb_$m")"
done
echo "LOAD DONE"

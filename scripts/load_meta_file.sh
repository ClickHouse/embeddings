#!/bin/bash
# Load metadata from the dumped TSV file into Cloud (regular-file stdin, server-side transform).
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
./clickhouse-new client --host "$H" --secure --user default --password "$PW" \
  --max_insert_block_size 200000 --query "
INSERT INTO mmcommons.yfcc_metadata
SELECT toUInt64OrZero(c1),c2,c3,parseDateTime64BestEffortOrNull(c4),toDateTime(toUInt32OrZero(c5)),c6,c7,c8,
 arrayFilter(x->x!='',arrayMap(x->decodeURLFormComponent(x),splitByChar(',',c9))),
 arrayFilter(x->x!='',arrayMap(x->decodeURLFormComponent(x),splitByChar(',',c10))),
 toFloat64OrNull(c11),toFloat64OrNull(c12),toUInt8OrZero(c13),c14,c15,c16,c17,toUInt32OrZero(c18),toUInt16OrZero(c19),c20,c21,c22,
 multiIf(c23='1','video','photo')
FROM input('c1 String,c2 String,c3 String,c4 String,c5 String,c6 String,c7 String,c8 String,c9 String,c10 String,c11 String,c12 String,c13 String,c14 String,c15 String,c16 String,c17 String,c18 String,c19 String,c20 String,c21 String,c22 String,c23 String') FORMAT TSV" < mmcommons/meta.tsv
echo "FILE_LOAD_RC=$? at $(date -u +%H:%M:%S)" > mmcommons/meta_fileload.done

#!/bin/bash
# Metadata load in fixed photo_id-range chunks: PK range-scan reads only the slice and flushes at EOF.
# Resumable: skips ranges already below the Cloud table's max photo_id.
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
DB=mmcommons/metadata/yfcc100m_dataset.sql
COLS="photoid,uid,unickname,datetaken,dateuploaded,capturedevice,title,description,usertags,machinetags,longitude,latitude,accuracy,pageurl,downloadurl,licensename,licenseurl,serverid,farmid,secret,secretoriginal,ext,marker"
CCQ(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" --query "$1"; }
MAXID=14068203843
W=50000000                      # photo_id range width per chunk (~281 chunks)
last=$(CCQ "SELECT ifNull(max(photo_id),0) FROM mmcommons.yfcc_metadata")
lo=$(( (last / W) * W ))        # resume at the range boundary at/just below last loaded id
echo "START maxid=$MAXID W=$W resume_lo=$lo $(date -u +%H:%M:%S)" >> mmcommons/meta_chunks.log
while [ "$lo" -le "$MAXID" ]; do
  hi=$(( lo + W ))
  sqlite3 -separator $'\t' "$DB" "SELECT $COLS FROM yfcc100m_dataset WHERE photoid>$lo AND photoid<=$hi" \
   | ./clickhouse-new client --host "$H" --secure --user default --password "$PW" --query "
INSERT INTO mmcommons.yfcc_metadata
SELECT toUInt64OrZero(c1),c2,c3,parseDateTime64BestEffortOrNull(c4),toDateTime(toUInt32OrZero(c5)),c6,c7,c8,
 arrayFilter(x->x!='',arrayMap(x->decodeURLFormComponent(x),splitByChar(',',c9))),
 arrayFilter(x->x!='',arrayMap(x->decodeURLFormComponent(x),splitByChar(',',c10))),
 toFloat64OrNull(c11),toFloat64OrNull(c12),toUInt8OrZero(c13),c14,c15,c16,c17,toUInt32OrZero(c18),toUInt16OrZero(c19),c20,c21,c22,
 multiIf(c23='1','video','photo')
FROM input('c1 String,c2 String,c3 String,c4 String,c5 String,c6 String,c7 String,c8 String,c9 String,c10 String,c11 String,c12 String,c13 String,c14 String,c15 String,c16 String,c17 String,c18 String,c19 String,c20 String,c21 String,c22 String,c23 String') FORMAT TSV"
  rc=${PIPESTATUS[1]:-1}
  [ "$rc" -ne 0 ] && { echo "FAIL range $lo-$hi rc=$rc $(date -u +%H:%M:%S)" >> mmcommons/meta_chunks.log; sleep 10; continue; }
  lo=$hi
  echo "DONE upto $hi $(date -u +%H:%M:%S)" >> mmcommons/meta_chunks.log
done
echo "META_CHUNKED_DONE $(date -u +%H:%M:%S)" >> mmcommons/meta_chunks.log

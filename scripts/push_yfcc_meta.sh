#!/bin/bash
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
./clickhouse-new local --query "
INSERT INTO FUNCTION remoteSecure('$H:9440','mmcommons.yfcc_metadata','default','$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD')
SELECT
  toUInt64OrZero(toString(photoid)), uid, unickname,
  parseDateTime64BestEffortOrNull(datetaken),
  toDateTime(toUInt32OrZero(toString(dateuploaded))),
  capturedevice, title, description,
  arrayFilter(x -> x != '', arrayMap(x -> decodeURLFormComponent(x), splitByChar(',', usertags))),
  arrayFilter(x -> x != '', arrayMap(x -> decodeURLFormComponent(x), splitByChar(',', machinetags))),
  toFloat64OrNull(longitude), toFloat64OrNull(latitude), toUInt8OrZero(toString(accuracy)),
  pageurl, downloadurl, licensename, licenseurl,
  toUInt32OrZero(toString(serverid)), toUInt16OrZero(toString(farmid)),
  secret, secretoriginal, ext,
  multiIf(toString(marker) = '1', 'video', 'photo')
FROM sqlite('mmcommons/metadata/yfcc100m_dataset.sql','yfcc100m_dataset')
SETTINGS max_execution_time = 0, max_block_size = 100000"
echo "META_PUSH_RC=$? at $(date -u +%H:%M:%S)" > mmcommons/meta_push.done

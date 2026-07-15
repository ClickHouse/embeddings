#!/bin/bash
cd /home/ubuntu/embeddings
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
PW="$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD"
ARGS=$(python3 -c "print(','.join(f'nemb[{i}]' for i in range(1,4097)))")
CC(){ ./clickhouse-new client --host "$H" --secure --user default --password "$PW" --query "$1" 2>&1 | grep -v "Unknown settings"; }
# method lr l2 batch epochs
for cfg in "SGD 0.5 0.01 16 30" "SGD 1.0 0.05 16 30" "Adam 0.005 0.001 16 30" "Adam 0.001 0.0001 16 40" "Adam 0.01 0.01 16 25"; do
  set -- $cfg; m=$1; lr=$2; l2=$3; b=$4; ep=$5
  CC "DROP TABLE IF EXISTS default.hn_bin_model" >/dev/null 2>&1
  ts=$(date +%s)
  CC "CREATE TABLE default.hn_bin_model ENGINE=MergeTree ORDER BY tuple() AS
      SELECT stochasticLogisticRegressionState($lr,$l2,$b,'$m')(h.y,$ARGS) AS state
      FROM numbers($ep) AS n CROSS JOIN default.hn_bin AS h WHERE sipHash64(h.id)%4!=0" >/dev/null 2>&1
  tr=$(( $(date +%s)-ts ))
  r=$(CC "SELECT round(avg((p>0.5)=y),4), round(avg(p),3), round(stddevPop(p),3) FROM (SELECT evalMLMethod((SELECT state FROM default.hn_bin_model),$ARGS) p, y FROM default.hn_bin WHERE sipHash64(id)%4=0) FORMAT TSV")
  echo "m=$m lr=$lr l2=$l2 b=$b ep=$ep -> acc/avgP/sdP = $r  train=${tr}s"
done
echo GRIDDONE
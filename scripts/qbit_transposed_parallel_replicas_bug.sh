#!/usr/bin/env bash
# Repro for ClickHouse/ClickHouse: cosineDistanceTransposed (BFloat16 QBit) with a
# scalar-subquery reference argument fails with Code 10 NOT_FOUND_COLUMN_IN_BLOCK when
# parallel replicas are enabled AND actually fan out. The coordinator asks for a
# `_CAST(__getScalar('<hash>'), 'Array(BFloat16)')` column while the remote block holds a
# materialized-constant column with a different name -> the two don't match.
#
# - cosineDistanceTransposedQuantized (Int8) is NOT affected.
# - Without parallel replicas it works.
# - Needs the scan to actually distribute across >=2 replicas, so use a multi-stride-group
#   QBit (dim 1152 / stride 128 -> 9 groups) and a small index_granularity so there are
#   enough marks to split. Requires a multi-replica cluster (on ClickHouse Cloud: 'default').
set -e
CH=(clickhouse client --allow_experimental_qbit_type 1)   # adjust connection flags
q(){ "${CH[@]}" --query "$1" 2>&1; }
PR="enable_parallel_replicas=1, automatic_parallel_replicas_mode=0, max_parallel_replicas=999, cluster_for_parallel_replicas='default', parallel_replicas_min_number_of_rows_per_replica=0, allow_experimental_qbit_type=1"

q "DROP DATABASE IF EXISTS prbug"
q "CREATE DATABASE prbug"
q "CREATE TABLE prbug.t (id UInt32, v Array(BFloat16),
      qb QBit(BFloat16,1152,128) DEFAULT CAST(v,'Array(BFloat16)'),
      qi QBit(Int8,1152,128)     DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(v,'Array(BFloat16)')))
    ENGINE = MergeTree ORDER BY id SETTINGS index_granularity = 128"
q "INSERT INTO prbug.t (id, v)
    SELECT number, arrayMap(i -> toFloat32(rand(i*number+1))/4294967296, range(1152)) FROM numbers(500000)"
q "ALTER TABLE prbug.t MATERIALIZE COLUMN qb, MATERIALIZE COLUMN qi SETTINGS mutations_sync = 1"

echo "A) BFloat16 cosineDistanceTransposed + scalar subquery + parallel replicas  -> BUG (Code 10):"
q "WITH (SELECT CAST(qb,'Array(Float32)') FROM prbug.t WHERE id=0) AS ref
   SELECT count() FROM (SELECT id FROM prbug.t ORDER BY cosineDistanceTransposed(qb, ref, 8, 1152) LIMIT 10)
   SETTINGS $PR" | grep -oiE "Code: [0-9]+.*NOT_FOUND_COLUMN_IN_BLOCK|Code: [0-9]+|^[0-9]+$" | head -1

echo "B) same, NO parallel replicas (control -> works):"
q "WITH (SELECT CAST(qb,'Array(Float32)') FROM prbug.t WHERE id=0) AS ref
   SELECT count() FROM (SELECT id FROM prbug.t ORDER BY cosineDistanceTransposed(qb, ref, 8, 1152) LIMIT 10)
   SETTINGS allow_experimental_qbit_type=1" | tail -1

echo "C) Int8 cosineDistanceTransposedQuantized + scalar subquery + parallel replicas (control -> works):"
q "WITH (SELECT CAST(qi,'Array(Float32)') FROM prbug.t WHERE id=0) AS ref
   SELECT count() FROM (SELECT id FROM prbug.t ORDER BY cosineDistanceTransposedQuantized(qi, ref, 8, 1152) LIMIT 10)
   SETTINGS $PR" | tail -1

q "DROP DATABASE IF EXISTS prbug"

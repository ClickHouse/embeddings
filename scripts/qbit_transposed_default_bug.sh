#!/usr/bin/env bash
# Repro: cosineDistanceTransposed / cosineDistanceTransposedQuantized read the source
# column as EMPTY (=> QBit CAST fails "got 0") when they read a not-yet-materialized
# QBit DEFAULT column WHILE a MATERIALIZE COLUMN mutation for it is in progress.
# Normal reads of the same column recompute the DEFAULT correctly the whole time.
set -e
CH=(clickhouse client --allow_experimental_qbit_type 1)   # adjust connection flags
q(){ "${CH[@]}" --query "$1" 2>&1; }
q "DROP DATABASE IF EXISTS qbit_default_bug"
q "CREATE DATABASE qbit_default_bug"
q "CREATE TABLE qbit_default_bug.t (id UInt32, v Array(BFloat16))
     ENGINE = MergeTree ORDER BY id SETTINGS min_bytes_for_wide_part = 0"
q "INSERT INTO qbit_default_bug.t
     SELECT number, arrayMap(i -> toFloat32(rand(i*number+1))/4294967296, range(128)) FROM numbers(6000000)"
q "ALTER TABLE qbit_default_bug.t
     ADD COLUMN qb QBit(BFloat16, 128, 128) DEFAULT CAST(v, 'Array(BFloat16)'),
     ADD COLUMN qi QBit(Int8, 128, 128) DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(v, 'Array(BFloat16)'))"
# Start the backfill mutation asynchronously.
q "ALTER TABLE qbit_default_bug.t MATERIALIZE COLUMN qb, MATERIALIZE COLUMN qi SETTINGS mutations_sync = 0"
NORM="SELECT sum(length(CAST(qb,'Array(BFloat16)'))) FROM qbit_default_bug.t WHERE id < 1000"
BF16="WITH (SELECT CAST(qb,'Array(Float32)') FROM qbit_default_bug.t WHERE id=0) AS r
      SELECT count() FROM qbit_default_bug.t WHERE cosineDistanceTransposed(qb, r, 16, 128) < 0.5"
INT8="WITH (SELECT CAST(qi,'Array(Float32)') FROM qbit_default_bug.t WHERE id=0) AS r
      SELECT count() FROM qbit_default_bug.t WHERE cosineDistanceTransposedQuantized(qi, r, 8, 128) < 0.5"
for i in $(seq 1 30); do
  todo=$(q "SELECT sum(parts_to_do) FROM system.mutations WHERE database='qbit_default_bug' AND NOT is_done")
  [ -z "$todo" ] || [ "$todo" = "0" ] && { echo "mutation finished (no repro this run)"; break; }
  st(){ echo "$1" | grep -qi 'code:' && echo "FAIL: $(echo "$1"|grep -oim1 'Code:.*got [0-9]*')" || echo "ok($1)"; }
  echo "parts_to_do=$todo | normal=$(st "$(q "$NORM")") | transposed_bf16=$(st "$(q "$BF16")") | transposed_int8=$(st "$(q "$INT8")")"
done

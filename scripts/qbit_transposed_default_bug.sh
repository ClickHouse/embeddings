#!/usr/bin/env bash
# Repro for ClickHouse/ClickHouse#110634
#
# cosineDistanceTransposed / cosineDistanceTransposedQuantized fail (Code 190
# SIZES_OF_ARRAYS_DONT_MATCH, "got 0") when reading a NON-MATERIALIZED QBit column
# that has a DEFAULT expression: the transposed read path feeds the DEFAULT's source
# column an empty array [] instead of recomputing it. A normal (non-transposed) read
# of the same column recomputes the DEFAULT correctly, and the transposed query works
# once the column is materialized. No mutation / concurrency is required.
#
# (An in-flight ALTER ... MATERIALIZE COLUMN mutation is just one way to have a
#  non-materialized column; see the tail of this script for that variant.)
set -e
CH=(clickhouse client --allow_experimental_qbit_type 1)   # adjust connection flags
q(){ "${CH[@]}" --query "$1" 2>&1; }

echo "=== deterministic repro: non-materialized QBit DEFAULT column (no mutation) ==="
q "DROP DATABASE IF EXISTS qbit_default_bug"
q "CREATE DATABASE qbit_default_bug"
q "CREATE TABLE qbit_default_bug.t (id UInt32, v Array(BFloat16))
     ENGINE = MergeTree ORDER BY id SETTINGS min_bytes_for_wide_part = 0"
q "INSERT INTO qbit_default_bug.t
     SELECT number, arrayMap(i -> toFloat32(rand(i*number+1))/4294967296, range(128)) FROM numbers(1000)"
# QBit columns with a DEFAULT, added AFTER the data -> not materialized on the existing part.
q "ALTER TABLE qbit_default_bug.t
     ADD COLUMN qb QBit(BFloat16, 128, 128) DEFAULT CAST(v, 'Array(BFloat16)'),
     ADD COLUMN qi QBit(Int8, 128, 128) DEFAULT arrayMap(quantizeBFloat16ToInt8, CAST(v, 'Array(BFloat16)'))"
st(){ echo "$1" | grep -qi 'code:' && echo "FAIL: $(echo "$1"|grep -oim1 'Code:.*got [0-9]*')" || echo "ok($1)"; }
NORM="SELECT sum(length(CAST(qb,'Array(BFloat16)'))) FROM qbit_default_bug.t"
BF16="WITH (SELECT CAST(qb,'Array(Float32)') FROM qbit_default_bug.t WHERE id=0) AS r
      SELECT count() FROM qbit_default_bug.t WHERE cosineDistanceTransposed(qb, r, 16, 128) < 0.5"
INT8="WITH (SELECT CAST(qi,'Array(Float32)') FROM qbit_default_bug.t WHERE id=0) AS r
      SELECT count() FROM qbit_default_bug.t WHERE cosineDistanceTransposedQuantized(qi, r, 8, 128) < 0.5"
echo "  normal read (recomputes DEFAULT):  $(st "$(q "$NORM")")     # expect ok(128000)"
echo "  cosineDistanceTransposed:          $(st "$(q "$BF16")")     # BUG: expect FAIL got 0"
echo "  cosineDistanceTransposedQuantized: $(st "$(q "$INT8")")     # BUG: expect FAIL got 0"
q "ALTER TABLE qbit_default_bug.t MATERIALIZE COLUMN qb, MATERIALIZE COLUMN qi SETTINGS mutations_sync = 1"
echo "  after MATERIALIZE COLUMN:"
echo "    cosineDistanceTransposed:          $(st "$(q "$BF16")")   # expect ok"
echo "    cosineDistanceTransposedQuantized: $(st "$(q "$INT8")")   # expect ok"

echo
echo "=== variant: same failure for the entire duration of an in-flight MATERIALIZE mutation ==="
q "DROP DATABASE IF EXISTS qbit_default_bug"
q "CREATE DATABASE qbit_default_bug"
q "CREATE TABLE qbit_default_bug.t (id UInt32, v Array(BFloat16))
     ENGINE = MergeTree ORDER BY id SETTINGS min_bytes_for_wide_part = 0"
q "INSERT INTO qbit_default_bug.t
     SELECT number, arrayMap(i -> toFloat32(rand(i*number+1))/4294967296, range(128)) FROM numbers(6000000)"
q "ALTER TABLE qbit_default_bug.t ADD COLUMN qb QBit(BFloat16, 128, 128) DEFAULT CAST(v, 'Array(BFloat16)')"
q "ALTER TABLE qbit_default_bug.t MATERIALIZE COLUMN qb SETTINGS mutations_sync = 0"
BF16b="WITH (SELECT CAST(qb,'Array(Float32)') FROM qbit_default_bug.t WHERE id=0) AS r
       SELECT count() FROM qbit_default_bug.t WHERE cosineDistanceTransposed(qb, r, 16, 128) < 0.5"
for i in $(seq 1 20); do
  todo=$(q "SELECT sum(parts_to_do) FROM system.mutations WHERE database='qbit_default_bug' AND NOT is_done")
  [ -z "$todo" ] || [ "$todo" = "0" ] && { echo "  mutation finished -> query now: $(st "$(q "$BF16b")")"; break; }
  echo "  parts_to_do=$todo | transposed=$(st "$(q "$BF16b")")"
done
q "DROP DATABASE IF EXISTS qbit_default_bug"

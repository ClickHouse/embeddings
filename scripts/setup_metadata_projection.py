#!/usr/bin/env python3
# Make author/license lookups on mmcommons.yfcc_metadata efficient for the Explorer preview.
# The umap tables key images by the compact "stupidHex" md5 (per-byte leading-zero strip); yfcc_metadata
# stores the full 32-char hex md5 and is sorted by photo_id, so a lookup by stupidHex md5 is a full scan.
# Add a MATERIALIZED md5_stupid column + an INDEX-ONLY projection (SELECT _part_offset ORDER BY md5_stupid),
# which prunes the lookup to a single granule via projection filtering (optimize_use_projection_filtering,
# on by default) WITHOUT duplicating the value columns — much smaller than a value-storing projection.
#   python3 scripts/setup_metadata_projection.py
# Idempotent: skips column materialization if md5_stupid already exists; always (re)builds the projection.
import os, http.client, ssl, time
PW = os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
T = 'mmcommons.yfcc_metadata'
# stupidHex(md5): for each of 16 bytes, drop a leading '0' nibble -> matches the umap tables' md5
STUPID = ("arrayStringConcat(arrayMap(i->if(substring(md5,i*2-1,1)='0',"
          "substring(md5,i*2,1),substring(md5,i*2-1,2)),range(1,17)))")

def q(sql, settings=''):
    c = http.client.HTTPSConnection('hvdvsqo23t.us-east-2.aws.clickhouse-staging.com', 8443,
                                    context=ssl.create_default_context(), timeout=3600)
    c.request('POST', '/?' + settings, body=sql.encode(),
              headers={'X-ClickHouse-User': 'default', 'X-ClickHouse-Key': PW})
    r = c.getresponse(); d = r.read().decode(); c.close()
    if r.status != 200: raise SystemExit(f"ERR {r.status}: {d[:500]}\n-- SQL: {sql[:200]}")
    return d.strip()

def wait():
    while True:
        n = q(f"SELECT count() FROM clusterAllReplicas(default,system.mutations) "
              f"WHERE table='yfcc_metadata' AND is_done=0")
        if n == '0': return
        time.sleep(5)

t0 = time.time()
# _part_offset in a projection requires this MergeTree setting
q(f"ALTER TABLE {T} MODIFY SETTING allow_part_offset_column_in_projections = 1")

has_col = q(f"SELECT count() FROM system.columns WHERE database='mmcommons' AND table='yfcc_metadata' AND name='md5_stupid'") != '0'
if not has_col:
    print("ADD + MATERIALIZE COLUMN md5_stupid (MATERIALIZED)", flush=True)
    q(f"ALTER TABLE {T} ADD COLUMN md5_stupid String MATERIALIZED {STUPID}", 'mutations_sync=1'); wait()
    q(f"ALTER TABLE {T} MATERIALIZE COLUMN md5_stupid", 'mutations_sync=1'); wait()
else:
    print("md5_stupid column already present — skipping materialization", flush=True)

print(f"(RE)BUILD index-only projection proj_md5  (+{time.time()-t0:.0f}s)", flush=True)
q(f"ALTER TABLE {T} DROP PROJECTION IF EXISTS proj_md5", 'mutations_sync=1'); wait()
q(f"ALTER TABLE {T} ADD PROJECTION proj_md5 (SELECT _part_offset ORDER BY md5_stupid)", 'mutations_sync=1'); wait()
q(f"ALTER TABLE {T} MATERIALIZE PROJECTION proj_md5", 'mutations_sync=1'); wait()

# verify: the lookup is pruned by projection filtering (read_rows tiny) and EXPLAIN projections=1 shows it.
# (EXPLAIN indexes=1 does NOT reflect projection filtering — see ClickHouse#110947.)
sample = q(f"SELECT md5_stupid FROM {T} LIMIT 1 OFFSET 555")
def read_rows(flt):
    c = http.client.HTTPSConnection('hvdvsqo23t.us-east-2.aws.clickhouse-staging.com', 8443,
                                    context=ssl.create_default_context(), timeout=600)
    sql = (f"SELECT user_nickname, user_nsid, license_name, license_url FROM {T} "
           f"WHERE md5_stupid = '{sample}' FORMAT Null")
    c.request('POST', f'/?optimize_use_projection_filtering={flt}', body=sql.encode(),
              headers={'X-ClickHouse-User': 'default', 'X-ClickHouse-Key': PW})
    r = c.getresponse(); r.read(); import json; s = json.loads(r.getheader('X-ClickHouse-Summary')); c.close()
    return int(s['read_rows'])

on, off = read_rows(1), read_rows(0)
plan = q(f"EXPLAIN projections=1 SELECT user_nickname, license_name FROM {T} WHERE md5_stupid = '{sample}'")
row = q(f"SELECT user_nickname, license_name FROM {T} WHERE md5_stupid = '{sample}' LIMIT 1 FORMAT TSV")
print(f"DONE in {time.time()-t0:.0f}s | read_rows filtering on={on:,} off={off:,} | "
      f"proj_in_plan={'proj_md5' in plan} | sample: {row}", flush=True)
if on >= off: print("WARNING: projection filtering did not reduce read_rows:\n" + plan, flush=True)

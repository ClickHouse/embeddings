#!/usr/bin/env python3
# Make author/license lookups on mmcommons.yfcc_metadata efficient for the Explorer preview.
# The umap tables key images by the compact "stupidHex" md5 (per-byte leading-zero strip); yfcc_metadata
# stores the full 32-char hex md5 and is sorted by photo_id, so a lookup by stupidHex md5 is a full scan.
# Add a MATERIALIZED md5_stupid column + a projection ordered by it, so the lookup is a binary search.
#   python3 scripts/setup_metadata_projection.py
# NB: heavy one-time mutation on 100M rows; each step waits for its mutation to finish before the next.
import os, http.client, ssl, time
PW = os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
T = 'mmcommons.yfcc_metadata'
# stupidHex(md5): for each of 16 bytes, drop a leading '0' nibble -> matches the umap tables' md5
STUPID = ("arrayStringConcat(arrayMap(i->if(substring(md5,i*2-1,1)='0',"
          "substring(md5,i*2,1),substring(md5,i*2-1,2)),range(1,17)))")
PROJ_COLS = "md5_stupid, user_nickname, user_nsid, license_name, license_url"

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
print("1/4 ADD COLUMN md5_stupid (MATERIALIZED)", flush=True)
q(f"ALTER TABLE {T} ADD COLUMN IF NOT EXISTS md5_stupid String MATERIALIZED {STUPID}", 'mutations_sync=1'); wait()
print(f"2/4 MATERIALIZE COLUMN md5_stupid  (+{time.time()-t0:.0f}s)", flush=True)
q(f"ALTER TABLE {T} MATERIALIZE COLUMN md5_stupid", 'mutations_sync=1'); wait()
print(f"3/4 ADD PROJECTION proj_md5  (+{time.time()-t0:.0f}s)", flush=True)
q(f"ALTER TABLE {T} ADD PROJECTION IF NOT EXISTS proj_md5 (SELECT {PROJ_COLS} ORDER BY md5_stupid)", 'mutations_sync=1'); wait()
print(f"4/4 MATERIALIZE PROJECTION proj_md5  (+{time.time()-t0:.0f}s)", flush=True)
q(f"ALTER TABLE {T} MATERIALIZE PROJECTION proj_md5", 'mutations_sync=1'); wait()

# verify: a lookup uses the projection (binary search, few granules)
sample = q(f"SELECT md5_stupid FROM {T} LIMIT 1 OFFSET 555")
plan = q(f"EXPLAIN indexes=1 SELECT user_nickname, license_name, license_url FROM {T} "
         f"WHERE md5_stupid = '{sample}'")
used = 'proj_md5' in plan
row = q(f"SELECT user_nickname, license_name FROM {T} WHERE md5_stupid = '{sample}' LIMIT 1 FORMAT TSV")
print(f"DONE in {time.time()-t0:.0f}s | projection_used={used} | sample lookup: {row}", flush=True)
if not used: print("WARNING: projection not used by the optimizer:\n" + plan, flush=True)

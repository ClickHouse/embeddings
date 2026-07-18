#!/usr/bin/env python3
# Add map coordinates to the umap_<ds>_2d tables so the Explorer can render the UMAP projection the
# same way as the linear one: mx,my (UInt32, 0..2^32) from x,y and mz (UInt16 hue) from `color`,
# each normalized by precomputed q01/q99 quantiles (clamped) so points fill the square map; plus a
# morton projection proj_umap(ORDER BY mortonEncode(mx,my)) for fast tile range queries.
#   python3 scripts/enrich_umap_2d.py [ds ...]   (default: clip nomic siglip2)
# NOTE: siglip2's UMAP is still loading -> its quantiles are PROVISIONAL (recompute + rematerialize
# when it finishes). clip/nomic are complete (full 99M) so theirs are final.
import os, http.client, ssl, time, sys
PW = os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
# ds -> (LOx,HIx, LOy,HIy, LOc,HIc)  = q01/q99 of x,y,color
QUANT = {
  'clip':    (-10.7572, 6.4833, -7.6871, 8.2627, -6.6626, 5.1027),
  'nomic':   (-6.8012,  9.857,  -7.8332, 6.1774, -5.6215, 6.8439),
  'siglip2': (-14.0801, 6.8807, -12.9129, 6.31,  -5.655,  6.4463),   # PROVISIONAL (26M, still loading)
}
def q(sql, settings=''):
    c=http.client.HTTPSConnection('hvdvsqo23t.us-east-2.aws.clickhouse-staging.com',8443,context=ssl.create_default_context(),timeout=1800)
    c.request('POST','/?'+settings, body=sql.encode(), headers={'X-ClickHouse-User':'default','X-ClickHouse-Key':PW})
    r=c.getresponse(); d=r.read().decode(); c.close()
    if r.status!=200: raise SystemExit(f"ERR {r.status}: {d[:400]}")
    return d.strip()
def wait_mut(tbl):
    while True:
        n=q(f"SELECT count() FROM clusterAllReplicas(default,system.mutations) WHERE table='{tbl}' AND is_done=0")
        if n=='0': return
        time.sleep(5)
def norm_u32(c,lo,hi): return f"toUInt32(round(clamp((toFloat64({c})-({lo}))/(({hi})-({lo})),0.,1.)*4294967295))"
def norm_u16(c,lo,hi): return f"toUInt16(round(clamp((toFloat64({c})-({lo}))/(({hi})-({lo})),0.,1.)*65535))"

for ds in (sys.argv[1:] or ['clip','nomic','siglip2']):
    lox,hix,loy,hiy,loc,hic = QUANT[ds]; t=f"umap_{ds}_2d"; t0=time.time()
    q(f"ALTER TABLE mmcommons.{t} DROP PROJECTION IF EXISTS proj_umap", 'mutations_sync=1'); wait_mut(t)
    q(f"ALTER TABLE mmcommons.{t} DROP COLUMN IF EXISTS mx, DROP COLUMN IF EXISTS my, DROP COLUMN IF EXISTS mz", 'mutations_sync=1'); wait_mut(t)
    q(f"""ALTER TABLE mmcommons.{t}
          ADD COLUMN mx UInt32 MATERIALIZED {norm_u32('x',lox,hix)},
          ADD COLUMN my UInt32 MATERIALIZED {norm_u32('y',loy,hiy)},
          ADD COLUMN mz UInt16 MATERIALIZED {norm_u16('color',loc,hic)}""")
    q(f"ALTER TABLE mmcommons.{t} MATERIALIZE COLUMN mx, MATERIALIZE COLUMN my, MATERIALIZE COLUMN mz", 'mutations_sync=1'); wait_mut(t)
    q(f"ALTER TABLE mmcommons.{t} ADD PROJECTION proj_umap (SELECT mx,my,mz,md5 ORDER BY mortonEncode(mx,my))")
    q(f"ALTER TABLE mmcommons.{t} MATERIALIZE PROJECTION proj_umap", 'mutations_sync=1'); wait_mut(t)
    rng=q(f"SELECT min(mx),max(mx),min(my),max(my) FROM mmcommons.{t}")
    parts=q(f"SELECT countIf(has(projections,'proj_umap'))=count() FROM system.parts WHERE database='mmcommons' AND table='{t}' AND active")
    print(f"{ds}: mx/my range {rng}, proj on all parts={parts}, {time.time()-t0:.0f}s", flush=True)
print("UMAP ENRICH DONE", flush=True)

#!/usr/bin/env python3
# Add map coordinates to the umap_<ds>_2d tables so the Explorer can render the UMAP projection the
# same way as the linear one: mx,my (UInt32, 0..2^32) from x,y and mz (UInt16 hue) from `color`,
# each normalized by quantiles (clamped) so points fill the square map; plus a morton projection
# proj_umap(ORDER BY mortonEncode(mx,my)) for fast tile range queries.
#   python3 scripts/enrich_umap_2d.py [ds ...]   (default: clip nomic siglip2)
# x,y use the q0.1 / q99.9 quantiles (was q1/q99, which clamped ~3% of points to the map edge -> tail
# clusters like the "cat cluster" appeared off-map; q0.1/q99.9 clamps only ~0.3%). color/hue keeps
# q1/q99 for contrast. Quantiles are computed live from the table (approximate TDigest).
# NOTE: siglip2's UMAP is still loading -> its quantiles are PROVISIONAL (recompute + rematerialize
# when it finishes). clip/nomic are complete (full 99M) so theirs are final.
import os, http.client, ssl, time, sys
PW = os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
XY_LO, XY_HI = 0.001, 0.999   # x,y spatial range (wider -> fewer edge-clamped tail clusters)
C_LO,  C_HI  = 0.01,  0.99    # color/hue range (tighter -> more hue contrast)
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
def quants(t):   # (LOx,HIx, LOy,HIy, LOc,HIc)
    r=q(f"SELECT quantile({XY_LO})(x),quantile({XY_HI})(x),quantile({XY_LO})(y),quantile({XY_HI})(y),"
        f"quantile({C_LO})(color),quantile({C_HI})(color) FROM mmcommons.{t}")
    return tuple(map(float, r.split('\t')))
def norm_u32(c,lo,hi): return f"toUInt32(round(clamp((toFloat64({c})-({lo}))/(({hi})-({lo})),0.,1.)*4294967295))"
def norm_u16(c,lo,hi): return f"toUInt16(round(clamp((toFloat64({c})-({lo}))/(({hi})-({lo})),0.,1.)*65535))"

for ds in (sys.argv[1:] or ['clip','nomic','siglip2']):
    t=f"umap_{ds}_2d"; t0=time.time()
    lox,hix,loy,hiy,loc,hic = quants(t)
    print(f"{ds}: quantiles x[{lox:.3f},{hix:.3f}] y[{loy:.3f},{hiy:.3f}] color[{loc:.3f},{hic:.3f}]", flush=True)
    q(f"ALTER TABLE mmcommons.{t} DROP PROJECTION IF EXISTS proj_umap", 'mutations_sync=1'); wait_mut(t)
    q(f"ALTER TABLE mmcommons.{t} DROP COLUMN IF EXISTS mx, DROP COLUMN IF EXISTS my, DROP COLUMN IF EXISTS mz", 'mutations_sync=1'); wait_mut(t)
    q(f"""ALTER TABLE mmcommons.{t}
          ADD COLUMN mx UInt32 MATERIALIZED {norm_u32('x',lox,hix)},
          ADD COLUMN my UInt32 MATERIALIZED {norm_u32('y',loy,hiy)},
          ADD COLUMN mz UInt16 MATERIALIZED {norm_u16('color',loc,hic)}""")
    q(f"ALTER TABLE mmcommons.{t} MATERIALIZE COLUMN mx, MATERIALIZE COLUMN my, MATERIALIZE COLUMN mz", 'mutations_sync=1'); wait_mut(t)
    q(f"ALTER TABLE mmcommons.{t} ADD PROJECTION proj_umap (SELECT mx,my,mz,md5 ORDER BY mortonEncode(mx,my))")
    q(f"ALTER TABLE mmcommons.{t} MATERIALIZE PROJECTION proj_umap", 'mutations_sync=1'); wait_mut(t)
    clamped=q(f"SELECT round(100*countIf(mx IN (0,4294967295) OR my IN (0,4294967295))/count(),2) FROM mmcommons.{t}")
    parts=q(f"SELECT countIf(has(projections,'proj_umap'))=count() FROM system.parts WHERE database='mmcommons' AND table='{t}' AND active")
    print(f"{ds}: edge-clamped={clamped}%, proj on all parts={parts}, {time.time()-t0:.0f}s", flush=True)
print("UMAP ENRICH DONE", flush=True)

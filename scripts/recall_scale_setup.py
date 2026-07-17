#!/usr/bin/env python3
# Build a larger recall-bench sample + exact ground truth, reusing the existing 20 queries.
#   python3 scripts/recall_scale_setup.py <label> <N>      e.g.  1m 1000000   |   10m 10000000
# Creates, per dataset:  bench.samp_<ds>_<label> (first N rows by md5, superset of the 100k sample)
#                        bench.gt_<ds>_<label>   (exact-cosine top-100 over that sample, per query)
# Reuses bench.q_<ds> (the same 20 query points — they live inside the first-100k sample, hence
# inside every larger first-N sample, so results are directly comparable across sample sizes).
import os, http.client, ssl, time, sys

HOST='hvdvsqo23t.us-east-2.aws.clickhouse-staging.com'; PORT=8443
PW=os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
LABEL = sys.argv[1] if len(sys.argv) > 1 else '1m'
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
# per-dataset QBit column types: (rotated, strided, int, rotated_int)
SPECS = {
 'emb_siglip2': ('QBit(BFloat16,2048,128)','QBit(BFloat16,1152,128)','QBit(Int8,1152,128)','QBit(Int8,2048,128)'),
 'emb_clip':    ('QBit(BFloat16,768,128)', 'QBit(BFloat16,768,128)', 'QBit(Int8,768,128)', 'QBit(Int8,768,128)'),
 'emb_nomic':   ('QBit(BFloat16,768,128)', 'QBit(BFloat16,768,128)', 'QBit(Int8,768,128)', 'QBit(Int8,768,128)'),
}
tls=ssl.create_default_context()
def q(sql, settings=''):
    c=http.client.HTTPSConnection(HOST,PORT,context=tls,timeout=3600)
    c.request('POST','/?allow_experimental_qbit_type=1'+settings, body=sql.encode(),
              headers={'X-ClickHouse-User':'default','X-ClickHouse-Key':PW})
    r=c.getresponse(); d=r.read().decode(); c.close()
    if r.status!=200: raise Exception(f"HTTP {r.status}: {d[:400]}\n--SQL: {sql[:160]}")
    return d.strip()

def setup(ds):
    rot,strd,intc,rintc = SPECS[ds]
    st=f"bench.samp_{ds}_{LABEL}"; gt=f"bench.gt_{ds}_{LABEL}"
    t0=time.time()
    cut=q(f"SELECT md5 FROM mmcommons.{ds} ORDER BY md5 LIMIT 1 OFFSET {N}")
    q(f"DROP TABLE IF EXISTS {st}"); q(f"DROP TABLE IF EXISTS {gt}")
    q(f"""CREATE TABLE {st} (md5 String, embedding Array(BFloat16),
          embedding_rotated {rot}, embedding_strided {strd}, embedding_int {intc}, embedding_rotated_int {rintc})
          ENGINE=MergeTree ORDER BY md5""")
    q(f"""INSERT INTO {st}
          SELECT md5, embedding, embedding_rotated, embedding_strided, embedding_int, embedding_rotated_int
          FROM mmcommons.{ds} WHERE md5 < '{cut}'""",
      settings='&max_memory_usage=48000000000&max_execution_time=3600&max_insert_threads=8')
    n=q(f"SELECT count() FROM {st}"); t1=time.time()
    q(f"CREATE TABLE {gt} (q_idx UInt32, gt10 Array(String), gt100 Array(String)) ENGINE=MergeTree ORDER BY q_idx")
    q(f"""INSERT INTO {gt}
          SELECT q_idx,
                 arraySlice(arrayMap(t->t.2, arraySort(t->t.1, groupArray((dist,md5)))),1,10),
                 arrayMap(t->t.2, arraySort(t->t.1, groupArray((dist,md5))))
          FROM (SELECT q.q_idx q_idx, s.md5 md5, cosineDistance(CAST(s.embedding,'Array(Float32)'), q.ref_orig) dist
                FROM {st} s CROSS JOIN bench.q_{ds} q WHERE s.md5 != q.q_md5
                ORDER BY q_idx, dist ASC LIMIT 100 BY q_idx) GROUP BY q_idx""",
      settings='&max_memory_usage=64000000000&max_execution_time=3600&max_bytes_before_external_sort=24000000000')
    g=q(f"SELECT count() FROM {gt}")
    print(f"{ds}: samp={n} rows (insert {t1-t0:.0f}s), gt={g} queries, total {time.time()-t0:.0f}s", flush=True)

print(f"Building label='{LABEL}' N={N:,} for {list(SPECS)}", flush=True)
for ds in SPECS:            # sequential: one big sort/insert on the cluster at a time
    setup(ds)
print("SETUP DONE", flush=True)

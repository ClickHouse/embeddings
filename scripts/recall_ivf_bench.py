#!/usr/bin/env python3
"""Recall of the IVF search path (Photos / siglip2) vs exact ground truth over the FULL table.

Unlike recall_codecs_bench.py this does not use a ~100k sample pool: the IVF clustering (16,384 centroids,
each row's `cid`) was built over all 99.2M rows, and a sample would leave most clusters nearly empty. So the
ground truth is an exact search over the whole of mmcommons.emb_siglip2_ivf.

Ground truth: for each of NQ query photos, the exact top-1000 by L2Distance over the full-precision vectors
(vector_search_use_quantized_codes = 0, no cluster filter). L2, not cosine, because that is what the IVF
query ranks by; the vectors are only approximately unit-length (norms 0.995-1.0), so the two orders differ
slightly. Query photos are the first NQ rows by cityHash64(md5) -- md5 is already a hash, this just makes
the pick deterministic. Each exact search reads ~230 GB (~12 s); the result is cached in GT_CACHE so
re-runs only redo the IVF grid.

Grid: nprobe x k x multiplier, using the exact SQL and settings the Explorer sends for IVF (index.html).
nprobe is the number of nearest centroids whose clusters are scanned; the Explorer fixes it at 64.
The centroids are found through their HNSW index, as in the Explorer. That index leaves ~2.5% of the centroids
unreachable (a centroid searched with its own vector does not come back), which costs ~4 points of recall@100;
use_skip_indexes = 0 would make the lookup exact at ~160 ms per query.

GUARD: the cluster filter reads the same rows whether or not the quantized codes are used, so read_rows
can't tell the paths apart here. Bytes per row can: measured ~234 B/row with codes vs ~2,166 B/row
full-precision (each vector is 2,304 B at BFloat16). The rescore of k x multiplier candidates also reads full
vectors, which lifts the codes path to ~1,100 B/row at nprobe 16 with 10,000 candidates -- still well under the
full-precision floor. Configs above QUANT_MAX_BPR are flagged as a full-precision fallback.

Runs as the read-only `website` user by default (no tables are created).
Usage:  [CH_USER=website CH_PASSWORD=] [NQ=100] python3 scripts/recall_ivf_bench.py
Writes: results/recall_ivf.csv
"""
import os, csv, json, time, ssl, threading, http.client, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get('CH_HOST', 'hvdvsqo23t.us-east-2.aws.clickhouse-staging.com')
PORT = int(os.environ.get('CH_PORT', '8443'))
USER = os.environ.get('CH_USER', 'website')
PW   = os.environ.get('CH_PASSWORD', '')
NQ   = int(os.environ.get('NQ', '100'))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_CACHE = os.environ.get('GT_CACHE', os.path.expanduser('~/.cache/embeddings/recall_ivf_gt.json'))

IVF, QTAB, CENT = 'mmcommons.emb_siglip2_ivf', 'mmcommons.emb_siglip2_rabitq', 'mmcommons.emb_siglip2_centroids'
DIM = 1152
NPROBES = (16, 32, 64, 128)
KS      = (10, 100, 1000)
MULTS   = (1, 5, 10)
QUANT_MAX_BPR = 1800

tls = ssl.create_default_context()
local = threading.local()

def post(sql, settings=None, fmt='TSV'):
    """POST a query; return (body, summary dict)."""
    qs = {'default_format': fmt, 'use_query_cache': '0', 'max_execution_time': '300'}
    qs.update(settings or {})
    path = '/?' + urllib.parse.urlencode(qs)
    for att in range(4):
        try:
            if not hasattr(local, 'c'):
                local.c = http.client.HTTPSConnection(HOST, PORT, context=tls, timeout=330)
            local.c.request('POST', path, body=sql.encode(),
                            headers={'X-ClickHouse-User': USER, 'X-ClickHouse-Key': PW})
            r = local.c.getresponse(); body = r.read().decode()
            if r.status != 200:
                raise Exception(body[:300])
            return body, json.loads(r.getheader('X-ClickHouse-Summary') or '{}')
        except Exception:
            try: local.c.close(); del local.c
            except Exception: pass
            if att == 3:
                raise
            time.sleep(2)

lines = lambda body: [l for l in body.strip().split('\n') if l]
QVEC = f"(SELECT embedding FROM {QTAB} WHERE md5 = {{q:String}} LIMIT 1)"

# ---- query photos + exact ground truth (cached) ------------------------------------------------
def exact_top1000(q_md5):
    body, _ = post(f"WITH {QVEC} AS qv SELECT md5 FROM {IVF} WHERE md5 != {{q:String}} "
                   f"ORDER BY L2Distance(embedding, qv) ASC LIMIT 1000",
                   {'param_q': q_md5, 'vector_search_use_quantized_codes': '0'})
    got = lines(body)
    assert len(got) == 1000, (q_md5, len(got))
    return got

gt = json.load(open(GT_CACHE)) if os.path.exists(GT_CACHE) else {}
if gt.get('nq') != NQ or len(gt.get('queries', [])) != NQ:
    body, _ = post(f"SELECT md5 FROM {IVF} ORDER BY cityHash64(md5) LIMIT {NQ}")
    gt = {'nq': NQ, 'queries': lines(body), 'top': {}}
todo = [q for q in gt['queries'] if q not in gt['top']]
print(f"ground truth: {NQ} queries, {len(todo)} to compute (~12 s each)", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as ex:
    for i, (q, top) in enumerate(zip(todo, ex.map(exact_top1000, todo)), 1):
        gt['top'][q] = top
        os.makedirs(os.path.dirname(GT_CACHE), exist_ok=True)
        json.dump(gt, open(GT_CACHE, 'w'))
        print(f"  [{i}/{len(todo)}] {q}  {time.time() - t0:.0f}s", flush=True)
GT = [(q, {k: set(gt['top'][q][:k]) for k in KS}) for q in gt['queries']]

body, _ = post(f"SELECT (SELECT count() FROM {IVF}), (SELECT count() FROM {CENT})")
TABLE_ROWS, NCLUSTERS = map(int, body.split())
print(f"table {TABLE_ROWS:,} rows, {NCLUSTERS:,} clusters", flush=True)

# ---- one IVF search: the Explorer's SQL and settings --------------------------------------------
def search(nprobe, k, mult, q_md5):
    sql = (f"WITH {QVEC} AS qv SELECT md5 FROM {IVF} "
           f"WHERE cid IN (SELECT toUInt32(cid) FROM {CENT} "
           f"ORDER BY L2Distance(vec, CAST(qv, 'Array(Float32)')) ASC LIMIT {nprobe}) "
           f"ORDER BY L2Distance(embedding, qv) ASC LIMIT {k + 1}")
    body, s = post(sql, {'param_q': q_md5,
                         'vector_search_use_quantized_codes': '1',
                         'vector_search_index_fetch_multiplier': str(mult),
                         'query_plan_max_limit_for_lazy_materialization': '10000',
                         'enable_parallel_replicas': '0', 'enable_filesystem_cache': '1',
                         'compatibility': '26.11', 'max_threads': '16'})
    # like the Explorer: no `md5 != query` filter, one extra row, and the query itself dropped here
    got = [m for m in lines(body) if m != q_md5][:k]
    return got, int(s.get('read_rows', 0)), int(s.get('read_bytes', 0))

def run(cfg):
    nprobe, k, mult = cfg
    rk = r10 = rows = byts = 0.0
    for q_md5, gts in GT:
        got, rr, rb = search(nprobe, k, mult, q_md5)
        rows += rr; byts += rb
        rk  += len(gts[k] & set(got)) / float(k)
        r10 += len(gts[10] & set(got[:10])) / 10.0
    n = len(GT)
    bpr = byts / max(rows, 1)
    return dict(nprobe=nprobe, clusters=NCLUSTERS, k=k, multiplier=mult, candidates=k * mult,
                dim=DIM, rows_read=round(rows / n), table_rows=TABLE_ROWS,
                read_frac=round(rows / n / TABLE_ROWS, 5), bytes_per_row=round(bpr, 1),
                quantized=int(bpr < QUANT_MAX_BPR),
                recall_at_k=round(rk / n, 4), recall10=round(r10 / n, 4))

configs = [(p, k, m) for p in NPROBES for k in KS for m in MULTS]
print(f"configs: {len(configs)}  ({len(configs) * len(GT)} searches)", flush=True)
results, t0 = [], time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for i, r in enumerate(ex.map(run, configs), 1):
        results.append(r)
        print(f"  [{i}/{len(configs)}] nprobe={r['nprobe']} k={r['k']} x{r['multiplier']}  "
              f"r@k={r['recall_at_k']:.3f} r@10={r['recall10']:.3f}  read {100 * r['read_frac']:.2f}%"
              f"{'' if r['quantized'] else '   *** FULL-PRECISION FALLBACK ***'}", flush=True)

COLS = ['nprobe', 'clusters', 'k', 'multiplier', 'candidates', 'dim', 'rows_read', 'table_rows',
        'read_frac', 'bytes_per_row', 'quantized', 'recall_at_k', 'recall10']
dest = os.path.join(BASE, 'results/recall_ivf.csv')
with open(dest, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in sorted(results, key=lambda r: (r['nprobe'], r['k'], r['multiplier'])):
        w.writerow(r)
bad = [r for r in results if not r['quantized']]
print(f"DONE {len(results)} configs in {time.time() - t0:.0f}s -> {dest}", flush=True)
if bad:
    print(f"WARNING: {len(bad)} config(s) did NOT use the quantized codes", flush=True)

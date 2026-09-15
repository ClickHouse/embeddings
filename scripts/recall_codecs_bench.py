#!/usr/bin/env python3
"""Recall of the Quantized-codec (RaBitQ / TurboQuant) search path vs exact-cosine ground truth.

Grid: 7 models x 2 codecs x 3 index-fetch multipliers = 42 configs, each averaged over the pool's query
points (100 by default). k is fixed at 100 so the result set lines up with the gt10/gt100 ground truth.

Ground truth comes from bench.qc_/gtc_<label>, which recall_codecs_setup.sh derives from the codec pool
using the same cosineDistance-over-two-BFloat16-arrays expression this harness issues. That matters: the
older gt_* tables cast to Array(Float32) first, and the precision gap made near-ties rank differently, so
the codec-disabled control scored 0.980 recall@10 on web/img_siglip2 where it must be exactly 1.000.

Why this can't reuse scripts/recall_bench.py: that harness scores every config with ONE query, a
CROSS JOIN over all 20 query points plus `LIMIT 100 BY q_idx`. That works for cosineDistanceTransposed
because it is a plain scalar function. The codec path is not -- it is a query-plan rewrite that only
fires on the bare `ORDER BY cosineDistance(col, <const>) LIMIT k` shape, and a CROSS JOIN is not that
shape. It would silently fall back to reading full-precision vectors and report recall ~1.0 with no
error. So each (config, query point) is its own query, and every response is checked (see GUARD below).

GUARD: read_rows from X-ClickHouse-Summary tells us which path actually ran. The two-stage codec scan
passes over the pool twice (codes for every row, then the candidates), so it reports ~2x the pool's row
count; a full-precision scan reports ~1x. Measured on a 100k pool: codes=1 -> 205,262 rows, codes=0 ->
106,170 rows. Note bytes/row is NOT a usable discriminator here -- the inflated row count divides the
same total bytes, so the codec path reads FEWER bytes/row than full precision on a small pool.
Additionally each model gets one control search with codes=0, which must score recall10 AND recall100
== 1.0 exactly; if it doesn't, the harness itself is wrong, not the codec.

Usage:  CH_PASSWORD=... [CH_USER=viewer] python3 scripts/recall_codecs_bench.py
Writes: results/recall_codecs.csv
"""
import os, csv, json, time, ssl, threading, http.client, urllib.parse
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get('CH_HOST', 'hvdvsqo23t.us-east-2.aws.clickhouse-staging.com')
PORT = int(os.environ.get('CH_PORT', '8443'))
USER = os.environ.get('CH_USER', 'viewer')
PW   = os.environ['CH_PASSWORD']
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K     = 100                 # top-k per search; matches the gt100 ground truth
MULTS = (1, 5, 10)          # vector_search_index_fetch_multiplier
CODECS = ('rabitq', 'turboquant')
CODE_BITS = {'rabitq': 1, 'turboquant': 2}      # bits per dimension, confirmed from measured bytes/row

# (corpus, dataset, label, dim) -- label names both the codec pools (bench.cs_<label>_<codec>) and the
# ground-truth tables (bench.qc_/gtc_<label>).
MODELS = [
    ('photos', 'siglip2',       'siglip2',           1152),
    ('photos', 'clip',          'clip',               768),
    ('photos', 'nomic',         'nomic',              768),
    ('web',    'img_siglip2',   'web_img_siglip2',   1152),
    ('web',    'txt_bge_large', 'web_txt_bge_large', 1024),
    ('hn',     'nomic',         'hn_nomic',           768),
    ('hn',     'bge_large',     'hn_bge_large',      1024),
]

tls = ssl.create_default_context()
local = threading.local()

def post(sql, settings=None, fmt='TSV'):
    """POST a query; return (body, read_rows, read_bytes)."""
    qs = {'default_format': fmt, 'use_query_cache': '0', 'max_execution_time': '300',
          'enable_parallel_replicas': '0', 'allow_experimental_qbit_type': '1'}
    qs.update(settings or {})
    path = '/?' + urllib.parse.urlencode(qs)
    for att in range(4):
        try:
            if not hasattr(local, 'c'):
                local.c = http.client.HTTPSConnection(HOST, PORT, context=tls, timeout=300)
            local.c.request('POST', path, body=sql.encode(),
                            headers={'X-ClickHouse-User': USER, 'X-ClickHouse-Key': PW})
            r = local.c.getresponse(); body = r.read().decode()
            if r.status != 200:
                raise Exception(body[:200])
            s = json.loads(r.getheader('X-ClickHouse-Summary') or '{}')
            return body, int(s.get('read_rows', 0)), int(s.get('read_bytes', 0))
        except Exception as e:
            try: local.c.close(); del local.c
            except Exception: pass
            if att == 3:
                raise
            time.sleep(1.5)

# ---- ground truth, fetched once per pool -------------------------------------------------------
print('loading ground truth...', flush=True)
GT = {}     # label -> [(q_md5, set(gt10), set(gt100)), ...] ordered by q_idx
POOL_ROWS = {}
for corpus, ds, label, dim in MODELS:
    body, _, _ = post(
        f"SELECT q.q_idx AS q_idx, q.q_md5 AS q_md5, g.gt10 AS gt10, g.gt100 AS gt100 "
        f"FROM bench.qc_{label} q JOIN bench.gtc_{label} g ON q.q_idx = g.q_idx ORDER BY q_idx",
        fmt='JSONEachRow')
    rows = [json.loads(l) for l in body.strip().split('\n') if l]
    GT[label] = [(r['q_md5'], set(r['gt10']), set(r['gt100'])) for r in rows]
    body, _, _ = post(f"SELECT count() FROM bench.cs_{label}_rabitq")
    POOL_ROWS[label] = int(body.strip())
    print(f"  {label}: {len(GT[label])} queries, pool {POOL_ROWS[label]:,}", flush=True)

# ---- one search --------------------------------------------------------------------------------
def search(label, codec, mult, q_md5, codes='1'):
    tbl = f"bench.cs_{label}_{codec}"
    sql = (f"WITH (SELECT embedding FROM {tbl} WHERE md5 = {{q:String}} LIMIT 1) AS qv "
           f"SELECT md5 FROM {tbl} WHERE md5 != {{q:String}} "
           f"ORDER BY cosineDistance(embedding, qv) ASC LIMIT {K}")
    body, rr, rb = post(sql, {'param_q': q_md5,
                              'vector_search_use_quantized_codes': codes,
                              'vector_search_index_fetch_multiplier': str(mult),
                              'query_plan_max_limit_for_lazy_materialization': '10000'})
    return [l for l in body.strip().split('\n') if l], rr, rb

def run(cfg):
    corpus, ds, label, dim, codec, mult = cfg
    r10 = r100 = r10in100 = 0.0
    bpr_sum = rows_sum = 0.0
    for q_md5, gt10, gt100 in GT[label]:
        got, rr, rb = search(label, codec, mult, q_md5)
        bpr_sum += rb / max(rr, 1); rows_sum += rr
        top10, top100 = set(got[:10]), set(got)
        r10      += len(gt10 & top10) / 10.0
        r100     += len(gt100 & top100) / float(K)
        r10in100 += len(gt10 & top100) / 10.0
    n = len(GT[label])
    bpr = bpr_sum / n
    code_bytes = dim * CODE_BITS[codec] // 8
    # two-stage codec scan passes over the pool ~twice; a full-precision fallback passes once
    quantized = 1 if rows_sum / n > 1.4 * POOL_ROWS[label] else 0
    return dict(corpus=corpus, dataset=ds, codec=codec, multiplier=mult, k=K, dim=dim,
                code_bytes=code_bytes, bytes_per_row=round(bpr, 1), quantized=quantized,
                recall10=round(r10 / n, 4), recall100=round(r100 / n, 4),
                recall10in100=round(r10in100 / n, 4))

# ---- control: codes=0 must reproduce the ground truth exactly, else the harness is wrong ------
print('control (codes=0, expect recall10 = recall100 = 1.0 over all query points):', flush=True)
ctrl_bad = 0
for corpus, ds, label, dim in MODELS:
    c10 = c100 = 0.0
    for q_md5, gt10, gt100 in GT[label]:
        got, _, _ = search(label, 'rabitq', 1, q_md5, codes='0')
        c10  += len(gt10 & set(got[:10])) / 10.0
        c100 += len(gt100 & set(got)) / float(K)
    nq = len(GT[label]); c10 /= nq; c100 /= nq
    bad = (c10 != 1.0 or c100 != 1.0)
    ctrl_bad += bad
    print(f"  {corpus}/{ds}: recall10={c10:.4f} recall100={c100:.4f}"
          f"{'   *** GROUND TRUTH MISMATCH ***' if bad else ''}", flush=True)
if ctrl_bad:
    raise SystemExit(f"aborting: {ctrl_bad} model(s) failed the exactness control")

configs = [(c, ds, label, dim, codec, mult)
           for c, ds, label, dim in MODELS for codec in CODECS for mult in MULTS]
print(f"configs: {len(configs)}  ({sum(len(GT[c[2]]) for c in configs)} searches)", flush=True)

results, done, t0 = [], 0, time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for res in ex.map(run, configs):
        results.append(res); done += 1
        print(f"  [{done}/{len(configs)}] {res['corpus']}/{res['dataset']} {res['codec']} "
              f"x{res['multiplier']}  r@100={res['recall100']:.3f}  {res['bytes_per_row']}B/row"
              f"{'' if res['quantized'] else '   *** FULL-PRECISION FALLBACK ***'}", flush=True)

COLS = ['corpus', 'dataset', 'codec', 'multiplier', 'k', 'dim', 'code_bytes', 'bytes_per_row',
        'quantized', 'recall10', 'recall100', 'recall10in100']
dest = os.path.join(BASE, 'results/recall_codecs.csv')
with open(dest, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in sorted(results, key=lambda r: (r['corpus'], r['dataset'], r['codec'], r['multiplier'])):
        w.writerow(r)

fell_back = [r for r in results if not r['quantized']]
print(f"DONE {len(results)} configs in {time.time() - t0:.0f}s -> {dest}", flush=True)
if fell_back:
    print(f"WARNING: {len(fell_back)} config(s) did NOT use the quantized codes -- their recall is "
          f"full-precision and must be discarded:", flush=True)
    for r in fell_back:
        print(f"  {r['corpus']}/{r['dataset']} {r['codec']} x{r['multiplier']} ({r['bytes_per_row']}B/row)", flush=True)

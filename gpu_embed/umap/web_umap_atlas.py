#!/usr/bin/env python
"""GPU UMAP atlas for WonderfulWeb embeddings (image + text) -> ClickHouse, keyed by url.

Same method as hn_umap_atlas / the mmcommons image umap_atlas: PCA-reduce to 64d (fit on a sample,
project all -> memmap + aligned url text file, cached), then cuML UMAP fit on a sample + transform all,
stream (url, coords) to ClickHouse. 2D=3-comp (x,y,color); 3D=4-comp (x,y,z,color).

--model is a web_emb_<model> table suffix, e.g. img_siglip2, txt_bge_large. Source + dest on the
embedding service (CH_HOST). Keyset pagination on url (String), like the original md5 version.
Env: CH_HOST, CH_PW, CH_USER. Pin GPU with CUDA_VISIBLE_DEVICES.
"""
import os, sys, argparse, time
import numpy as np

DIMS = {"img_siglip2": 1152, "img_clip": 768, "img_nomic": 768,
        "txt_bge_large": 1024, "txt_e5_large": 1024, "txt_jina_v3": 1024, "txt_nomic": 768}


def get_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], port=8443, secure=True,
        username=os.environ.get("CH_USER", "default"), password=os.environ["CH_PW"],
        settings={"max_execution_time": 0})


def ensure_tables(client, model):
    eng = "SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}')"
    client.command(f"CREATE TABLE IF NOT EXISTS web_umap_{model}_2d "
                   f"(url String, x Float32, y Float32, color Float32) ENGINE={eng} ORDER BY url")
    client.command(f"CREATE TABLE IF NOT EXISTS web_umap_{model}_3d "
                   f"(url String, x Float32, y Float32, z Float32, color Float32) ENGINE={eng} ORDER BY url")


def fetch_batch(client, model, last_url, batch):
    dim = DIMS[model]
    esc = last_url.replace("\\", "\\\\").replace("'", "\\'")
    t = client.query_arrow(
        f"SELECT url, CAST(embedding AS Array(Float32)) AS embedding FROM web_emb_{model} "
        f"WHERE url > '{esc}' ORDER BY url LIMIT {batch}")
    if t.num_rows == 0:
        return None, None, last_url
    urls = t.column("url").to_pylist()
    vals = t.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    arr = np.ascontiguousarray(vals, dtype=np.float32).reshape(len(urls), dim)
    return urls, arr, urls[-1]


def build_reduced(client, model, pca_dim, pca_sample, read_batch, work, limit):
    import cupy as cp
    from cuml.decomposition import PCA
    red_path = f"{work}/web_{model}_pca{pca_dim}.f32"
    url_path = f"{work}/web_{model}_pca{pca_dim}.urls"
    meta = f"{work}/web_{model}_pca{pca_dim}.meta"
    if os.path.exists(meta):
        N = int(open(meta).read().split()[0]); print(f"[{model}] reduced cache hit: N={N}", flush=True)
        return red_path, url_path, N
    lock = f"{work}/web_{model}_pca{pca_dim}.lock"
    while True:
        if os.path.exists(meta):
            N = int(open(meta).read().split()[0]); return red_path, url_path, N
        try:
            os.mkdir(lock); break
        except FileExistsError:
            time.sleep(10)

    budget = int(os.environ.get("PCA_FIT_BYTES", str(6 * 10**9)))
    max_sample = max(200_000, budget // (DIMS[model] * 4))
    pca_sample = min(pca_sample, max_sample)
    print(f"[{model}] fitting PCA({pca_dim}) on up to {pca_sample} rows...", flush=True)
    Xs, got, last = [], 0, ""
    while got < pca_sample:
        urls, arr, last = fetch_batch(client, model, last, min(read_batch, pca_sample - got))
        if urls is None:
            break
        Xs.append(arr); got += len(urls)
    Xs = np.concatenate(Xs)
    pca = PCA(n_components=pca_dim, output_type="numpy"); pca.fit(cp.asarray(Xs))
    print(f"[{model}] PCA fit on {len(Xs)}; explained var={float(pca.explained_variance_ratio_.sum()):.3f}", flush=True)

    total = client.query(f"SELECT count() FROM web_emb_{model}").result_rows[0][0]
    if limit:
        total = min(total, limit)
    red = np.memmap(red_path, dtype=np.float32, mode="w+", shape=(total, pca_dim))
    furl = open(url_path, "w"); off, last, t0 = 0, "", time.time()
    while off < total:
        urls, arr, last = fetch_batch(client, model, last, min(read_batch, total - off))
        if urls is None:
            break
        red[off:off + len(urls)] = pca.transform(cp.asarray(arr))
        furl.write("\n".join(urls) + "\n")
        off += len(urls)
        if off % (read_batch * 20) == 0:
            print(f"[{model}] projected {off}/{total} ({off/(time.time()-t0):.0f}/s)", flush=True)
    red.flush(); furl.close(); open(meta, "w").write(f"{off} {pca_dim}\n")
    try: os.rmdir(lock)
    except OSError: pass
    print(f"[{model}] reduced done: N={off} in {time.time()-t0:.0f}s", flush=True)
    return red_path, url_path, off


def insert_coords(client, table, urls, coords, has_z):
    import pyarrow as pa
    cols = {"url": pa.array(urls, pa.string()), "x": pa.array(coords[:, 0], pa.float32()),
            "y": pa.array(coords[:, 1], pa.float32())}
    if has_z:
        cols["z"] = pa.array(coords[:, 2], pa.float32()); cols["color"] = pa.array(coords[:, 3], pa.float32())
    else:
        cols["color"] = pa.array(coords[:, 2], pa.float32())
    client.insert_arrow(table, pa.table(cols))


def run_umap(client, model, comps, red_path, url_path, N, pca_dim, args):
    import cupy as cp
    from cuml.manifold import UMAP
    table = f"web_umap_{model}_{'3d' if comps >= 4 else '2d'}"; has_z = comps >= 4
    red = np.memmap(red_path, dtype=np.float32, mode="r", shape=(N, pca_dim))
    rng = np.random.default_rng(args.seed); k = min(args.fit_sample, N)
    idx = np.sort(rng.choice(N, size=k, replace=False))
    print(f"[{model}/{comps}c] fitting UMAP on {k} pts", flush=True)
    um = UMAP(n_components=comps, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
              build_algo="nn_descent", random_state=args.seed, output_type="numpy")
    t0 = time.time(); um.fit(cp.asarray(np.ascontiguousarray(red[idx])))
    print(f"[{model}/{comps}c] fit {time.time()-t0:.0f}s; transforming {N} -> {table}", flush=True)
    client.command(f"TRUNCATE TABLE {table}")
    uf = open(url_path); off, t1 = 0, time.time()
    while off < N:
        n = min(args.tf_batch, N - off)
        coords = np.asarray(um.transform(cp.asarray(np.ascontiguousarray(red[off:off + n]))))
        urls = [uf.readline().rstrip("\n") for _ in range(n)]
        insert_coords(client, table, urls, coords, has_z)
        off += n
        if off % (args.tf_batch * 10) == 0:
            print(f"[{model}/{comps}c] {off}/{N} ({off/(time.time()-t1):.0f}/s)", flush=True)
    uf.close(); print(f"[{model}/{comps}c] DONE {off} in {time.time()-t1:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DIMS))
    ap.add_argument("--dims", default="2d,3d")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-sample", type=int, default=1_000_000)
    ap.add_argument("--fit-sample", type=int, default=1_000_000)
    ap.add_argument("--read-batch", type=int, default=100_000)
    ap.add_argument("--tf-batch", type=int, default=500_000)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(__file__), "..", "umap_work"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reduce-only", action="store_true")
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True)
    client = get_client(); ensure_tables(client, a.model)
    red_path, url_path, N = build_reduced(client, a.model, a.pca_dim, a.pca_sample, a.read_batch, a.work, a.limit)
    if a.reduce_only:
        print(f"[{a.model}] reduce-only done N={N}", flush=True); return
    for d in [x.strip() for x in a.dims.split(",") if x.strip()]:
        run_umap(client, a.model, 4 if d == "3d" else 3, red_path, url_path, N, a.pca_dim, a)
    print(f"[{a.model}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

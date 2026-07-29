#!/usr/bin/env python
"""GPU UMAP atlas for HackerNews text embeddings -> ClickHouse.

Mirrors umap_atlas.py (the mmcommons image version) but keyed by `id` (UInt32) instead of md5,
sourcing default.hackernews_embeddings_<model> (column `embedding` QBit(BFloat16,dim)).

Per model:
  1. PCA-reduce embeddings to --pca-dim (default 64): fit PCA on a sample (id order == a fixed subset),
     project ALL rows -> float32 memmap on disk + an aligned uint32 `id` file (cached; re-runs skip this).
  2. Fit cuML UMAP on a --fit-sample subset of the reduced data, then TRANSFORM all points.
  3. Stream (id, coords) to ClickHouse.

2D -> 3-component UMAP (x,y + color); 3D -> 4-component (x,y,z + color). color = extra UMAP axis.
Output: hackernews_umap_<model>_2d(id,x,y,color) / _3d(id,x,y,z,color), keyed by id (joins to hackernews_embeddings_*).

Deps: venv_umap (cuml-cu12, cupy, clickhouse-connect, numpy, pyarrow). Env: CH_HOST, CH_PW, CH_USER(=default).
Pin GPU with CUDA_VISIBLE_DEVICES.
"""
import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from text_models import MODELS
DIMS = {k: v["dim"] for k, v in MODELS.items()}
DIMS["bow"] = 8192   # bag-of-words (hashing trick), Array(BFloat16); source hackernews_embeddings_bow
DIMS["qwen3_8b"] = 4096   # external Qwen3-8B embeddings; source hackernews_embeddings_qwen3_8b (QBit BFloat16)


def get_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], port=8443, secure=True,
        username=os.environ.get("CH_USER", "default"), password=os.environ["CH_PW"],
        settings={"max_execution_time": 0})


def ensure_tables(client, model):
    eng = "SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}')"
    client.command(f"CREATE TABLE IF NOT EXISTS hackernews_umap_{model}_2d "
                   f"(id UInt32, x Float32, y Float32, color Float32) ENGINE={eng} ORDER BY id")
    client.command(f"CREATE TABLE IF NOT EXISTS hackernews_umap_{model}_3d "
                   f"(id UInt32, x Float32, y Float32, z Float32, color Float32) ENGINE={eng} ORDER BY id")


def fetch_batch(client, model, last_id, batch):
    """One page of (id, embedding) with id > last_id, in id order. -> (ids int list, float32[n,dim], new_last)."""
    dim = DIMS[model]
    t = client.query_arrow(
        f"SELECT id, CAST(embedding AS Array(Float32)) AS embedding FROM hackernews_embeddings_{model} "
        f"WHERE id > {last_id} ORDER BY id LIMIT {batch}")
    if t.num_rows == 0:
        return None, None, last_id
    ids = t.column("id").to_pylist()
    vals = t.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    arr = np.ascontiguousarray(vals, dtype=np.float32).reshape(len(ids), dim)
    return ids, arr, ids[-1]


def build_reduced(client, model, pca_dim, pca_sample, read_batch, work, limit, read_threads=1):
    """Fit PCA on a sample, project ALL rows -> memmap (N,pca_dim) + aligned uint32 id file. Cached via .meta."""
    import cupy as cp
    from cuml.decomposition import PCA
    red_path = f"{work}/hn_{model}_pca{pca_dim}.f32"
    id_path = f"{work}/hn_{model}_pca{pca_dim}.ids"
    meta = f"{work}/hn_{model}_pca{pca_dim}.meta"
    if os.path.exists(meta):
        N = int(open(meta).read().split()[0])
        print(f"[{model}] reduced cache hit: N={N}", flush=True)
        return red_path, id_path, N

    lock = f"{work}/hn_{model}_pca{pca_dim}.lock"
    while True:
        if os.path.exists(meta):
            N = int(open(meta).read().split()[0]); print(f"[{model}] reduced cache hit (waited): N={N}", flush=True)
            return red_path, id_path, N
        try:
            os.mkdir(lock); break
        except FileExistsError:
            time.sleep(10)

    budget = int(os.environ.get("PCA_FIT_BYTES", str(6 * 10**9)))
    max_sample = max(200_000, budget // (DIMS[model] * 4))
    if pca_sample > max_sample:
        print(f"[{model}] capping pca_sample {pca_sample} -> {max_sample} (dim={DIMS[model]}, GPU mem)", flush=True)
        pca_sample = max_sample

    print(f"[{model}] fitting PCA({pca_dim}) on up to {pca_sample} sample rows...", flush=True)
    Xs, got, last = [], 0, -1
    while got < pca_sample:
        ids, arr, last = fetch_batch(client, model, last, min(read_batch, pca_sample - got))
        if ids is None:
            break
        Xs.append(arr); got += len(ids)
    Xs = np.concatenate(Xs)
    pca = PCA(n_components=pca_dim, output_type="numpy")
    pca.fit(cp.asarray(Xs))
    print(f"[{model}] PCA fit on {len(Xs)} rows; explained var={float(pca.explained_variance_ratio_.sum()):.3f}", flush=True)

    total = client.query(f"SELECT count() FROM hackernews_embeddings_{model}").result_rows[0][0]
    if limit:
        total = min(total, limit)
    red = np.memmap(red_path, dtype=np.float32, mode="w+", shape=(total, pca_dim))
    fids = open(id_path, "wb")
    off, t0 = 0, time.time()
    if read_threads <= 1:
        last = -1
        while off < total:
            ids, arr, last = fetch_batch(client, model, last, min(read_batch, total - off))
            if ids is None:
                break
            red[off:off + len(ids)] = pca.transform(cp.asarray(arr))
            fids.write(np.asarray(ids, dtype=np.uint32).tobytes())
            off += len(ids)
            if off % (read_batch * 20) == 0:
                print(f"[{model}] projected {off}/{total} ({off/(time.time()-t0):.0f}/s)", flush=True)
    else:
        # Parallel sharded read (network-bound for high-dim like bow=8192): K reader threads each fetch an
        # id-range shard; the main thread PCA-projects (GPU, single-threaded) + writes. Row order need not be
        # sorted (UMAP fit samples by index; transform is order-agnostic) — ids stay aligned with vectors.
        from concurrent.futures import ThreadPoolExecutor
        lo0 = client.query(f"SELECT min(id) FROM hackernews_embeddings_{model}").result_rows[0][0]
        hi0 = client.query(f"SELECT max(id) FROM hackernews_embeddings_{model}").result_rows[0][0]
        S = read_threads * 40
        width = max(1, (hi0 - lo0 + 1 + S - 1) // S)
        shards = [(lo0 + i * width, min(lo0 + (i + 1) * width, hi0 + 1)) for i in range(S)]

        def read_shard(ab):
            a, b = ab
            c = get_client()
            t = c.query_arrow(f"SELECT id, CAST(embedding AS Array(Float32)) AS embedding "
                              f"FROM hackernews_embeddings_{model} WHERE id >= {a} AND id < {b} ORDER BY id")
            if t.num_rows == 0:
                return np.empty(0, np.uint32), np.empty((0, DIMS[model]), np.float32)
            ids = np.asarray(t.column("id").to_pylist(), dtype=np.uint32)
            vals = t.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
            return ids, np.ascontiguousarray(vals, dtype=np.float32).reshape(len(ids), DIMS[model])

        with ThreadPoolExecutor(max_workers=read_threads) as ex:
            for ids, arr in ex.map(read_shard, shards):
                if len(ids) == 0 or off >= total:
                    continue
                n = min(len(ids), total - off)          # respect limit / memmap bound
                red[off:off + n] = pca.transform(cp.asarray(arr[:n]))
                fids.write(ids[:n].tobytes())
                off += n
                if off % (read_batch * 20) < n:
                    print(f"[{model}] projected {off}/{total} ({off/(time.time()-t0):.0f}/s)", flush=True)
    red.flush(); fids.close()
    open(meta, "w").write(f"{off} {pca_dim}\n")
    try: os.rmdir(lock)
    except OSError: pass
    print(f"[{model}] reduced done: N={off} in {time.time()-t0:.0f}s", flush=True)
    return red_path, id_path, off


def insert_coords(client, table, ids, coords, has_z):
    import pyarrow as pa
    cols = {"id": pa.array(ids, pa.uint32()),
            "x": pa.array(coords[:, 0], pa.float32()),
            "y": pa.array(coords[:, 1], pa.float32())}
    if has_z:
        cols["z"] = pa.array(coords[:, 2], pa.float32())
        cols["color"] = pa.array(coords[:, 3], pa.float32())
    else:
        cols["color"] = pa.array(coords[:, 2], pa.float32())
    tbl = pa.table(cols)
    last = None
    for attempt in range(6):   # transient CH insert timeouts otherwise kill the whole job
        try:
            client.insert_arrow(table, tbl); return
        except Exception as e:
            last = e; time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"insert into {table} failed after retries: {last}")


def run_umap(client, model, comps, red_path, id_path, N, pca_dim, args):
    import cupy as cp
    from cuml.manifold import UMAP
    table = f"hackernews_umap_{model}_{'3d' if comps >= 4 else '2d'}"
    has_z = comps >= 4
    red = np.memmap(red_path, dtype=np.float32, mode="r", shape=(N, pca_dim))
    ids_all = np.memmap(id_path, dtype=np.uint32, mode="r", shape=(N,))

    rng = np.random.default_rng(args.seed)
    k = min(args.fit_sample, N)
    idx = np.sort(rng.choice(N, size=k, replace=False))
    print(f"[{model}/{comps}c] fitting UMAP on {k} pts (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})", flush=True)
    um = UMAP(n_components=comps, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
              build_algo="nn_descent", random_state=args.seed, output_type="numpy")
    t0 = time.time()
    um.fit(cp.asarray(np.ascontiguousarray(red[idx])))
    print(f"[{model}/{comps}c] UMAP fit in {time.time()-t0:.0f}s; transforming {N} pts -> {table}", flush=True)

    client.command(f"TRUNCATE TABLE {table}")
    off, t1 = 0, time.time()
    while off < N:
        n = min(args.tf_batch, N - off)
        coords = np.asarray(um.transform(cp.asarray(np.ascontiguousarray(red[off:off + n]))))
        ids_b = ids_all[off:off + n].tolist()
        insert_coords(client, table, ids_b, coords, has_z)
        off += n
        if off % (args.tf_batch * 10) == 0:
            print(f"[{model}/{comps}c] transformed {off}/{N} ({off/(time.time()-t1):.0f}/s)", flush=True)
    print(f"[{model}/{comps}c] DONE {off} pts in {time.time()-t1:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DIMS))
    ap.add_argument("--dims", default="2d,3d", help="which to build: 2d,3d")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-sample", type=int, default=2_000_000)
    ap.add_argument("--fit-sample", type=int, default=1_000_000)  # cuML UMAP transform cost ~ fit-set size;
    # 1M fits the 37.8M manifold well and transforms ~3x faster than 3M (~20k rows/s -> ~30min/job)
    ap.add_argument("--read-batch", type=int, default=200_000)
    ap.add_argument("--tf-batch", type=int, default=500_000)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(__file__), "..", "umap_work"))
    ap.add_argument("--limit", type=int, default=0, help="cap N for a quick validation run (0=all)")
    ap.add_argument("--reduce-only", action="store_true", help="build the PCA-reduced cache and exit (no UMAP)")
    ap.add_argument("--read-threads", type=int, default=1, help="parallel read shards for the projection (use 8 for high-dim like bow)")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    client = get_client()
    ensure_tables(client, args.model)
    red_path, id_path, N = build_reduced(client, args.model, args.pca_dim,
                                         args.pca_sample, args.read_batch, args.work, args.limit,
                                         read_threads=args.read_threads)
    if args.reduce_only:
        print(f"[{args.model}] reduce-only done: N={N}", flush=True)
        return
    for d in [x.strip() for x in args.dims.split(",") if x.strip()]:
        run_umap(client, args.model, 4 if d == "3d" else 3, red_path, id_path, N, args.pca_dim, args)
    print(f"[{args.model}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

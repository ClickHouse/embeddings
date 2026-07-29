#!/usr/bin/env python
"""GPU UMAP atlas for mmcommons embeddings -> ClickHouse.

Per model (nomic|clip|siglip2):
  1. PCA-reduce the embeddings to --pca-dim (default 64) so ~99.6M points fit in memory/GPU.
     PCA is FIT on a sample (md5 order == hash order == a uniform random sample), then ALL points are
     projected and streamed to a float32 memmap on disk (cached -> re-runs skip this).
  2. Fit cuML UMAP on a --fit-sample subset of the reduced data, then TRANSFORM all points.
  3. Stream (md5, coords) to ClickHouse.

Defaults chosen for the away-user (see run_umap.sh header):
  color = an extra UMAP dimension. 2D -> 3-component UMAP (x,y + color); 3D -> 4-component (x,y,z + color).
Runs both 2D and 3D for the model in one process (reuses the PCA/reduced data).

Deps: venv_umap (cuml-cu12, cupy-cuda12x, clickhouse-connect, numpy, pyarrow). See setup_umap.sh.
Env: CH_HOST, CH_PW, CH_USER(=default). Pin the GPU with CUDA_VISIBLE_DEVICES.
NOTE: not yet run end-to-end (needs RAPIDS + siglip2 done). Validate with --limit 200000 first.
"""
import os, sys, argparse, time
import numpy as np

DIMS = {"nomic": 768, "clip": 768, "siglip2": 1152}


def get_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], port=8443, secure=True,
        username=os.environ.get("CH_USER", "default"), password=os.environ["CH_PW"],
        settings={"max_execution_time": 0})


def ensure_tables(client, model):
    client.command(f"CREATE TABLE IF NOT EXISTS mmcommons.umap_{model}_2d "
                   f"(md5 String, x Float32, y Float32, color Float32) ENGINE=MergeTree ORDER BY md5")
    client.command(f"CREATE TABLE IF NOT EXISTS mmcommons.umap_{model}_3d "
                   f"(md5 String, x Float32, y Float32, z Float32, color Float32) ENGINE=MergeTree ORDER BY md5")


def fetch_batch(client, model, last_md5, batch):
    """One page of (md5, embedding) after last_md5, in md5 order. -> (md5_list, float32[n,dim], new_last)."""
    import pyarrow as pa
    dim = DIMS[model]
    # CAST bf16 -> Float32 in SQL: Arrow has no reliable BFloat16, so let CH emit float32 arrays.
    t = client.query_arrow(
        f"SELECT md5, CAST(embedding AS Array(Float32)) AS embedding FROM mmcommons.emb_{model} "
        f"WHERE md5 > '{last_md5}' ORDER BY md5 LIMIT {batch}")
    if t.num_rows == 0:
        return None, None, last_md5
    md5 = t.column("md5").to_pylist()
    # embedding is list<float> (CH casts BFloat16 -> float32 in Arrow); flatten child values and reshape
    vals = t.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    arr = np.ascontiguousarray(vals, dtype=np.float32).reshape(len(md5), dim)
    return md5, arr, md5[-1]


def build_reduced(client, model, pca_dim, pca_sample, read_batch, work, limit):
    """Fit PCA on a sample, project ALL rows -> memmap (N,pca_dim) + aligned md5 file. Cached via .meta."""
    import cupy as cp
    from cuml.decomposition import PCA
    red_path = f"{work}/{model}_pca{pca_dim}.f32"
    md5_path = f"{work}/{model}_pca{pca_dim}.md5"
    meta = f"{work}/{model}_pca{pca_dim}.meta"
    if os.path.exists(meta):
        N = int(open(meta).read().split()[0])
        print(f"[{model}] reduced cache hit: N={N}", flush=True)
        return red_path, md5_path, N

    # Only ONE process may build the reduce. Concurrent jobs for the same model (e.g. siglip2 2d & 3d,
    # if the phase-1 --reduce-only failed) would otherwise race on the shared red/md5 files. Winner builds
    # under a lock dir (atomic mkdir); others poll for the finished cache.
    lock = f"{work}/{model}_pca{pca_dim}.lock"
    while True:
        if os.path.exists(meta):
            N = int(open(meta).read().split()[0]); print(f"[{model}] reduced cache hit (waited): N={N}", flush=True)
            return red_path, md5_path, N
        try:
            os.mkdir(lock); break            # we won the lock -> build below
        except FileExistsError:
            time.sleep(10)

    # Cap the PCA fit-sample so it (x~2 for cuML's working copy) fits GPU memory. High-dim models
    # (siglip2 = 1152) OOM a 24GB L4 at the 3M default: 3M*1152*4 = 13.8GB *2 > 24GB. A ~1M sample is
    # plenty to estimate 64 principal components. Budget tunable via PCA_FIT_BYTES.
    budget = int(os.environ.get("PCA_FIT_BYTES", str(6 * 10**9)))
    max_sample = max(200_000, budget // (DIMS[model] * 4))
    if pca_sample > max_sample:
        print(f"[{model}] capping pca_sample {pca_sample} -> {max_sample} (dim={DIMS[model]}, GPU mem)", flush=True)
        pca_sample = max_sample

    print(f"[{model}] fitting PCA({pca_dim}) on up to {pca_sample} sample rows...", flush=True)
    Xs, got, last = [], 0, ""
    while got < pca_sample:
        md5, arr, last = fetch_batch(client, model, last, min(read_batch, pca_sample - got))
        if md5 is None:
            break
        Xs.append(arr); got += len(md5)
    Xs = np.concatenate(Xs)
    pca = PCA(n_components=pca_dim, output_type="numpy")
    pca.fit(cp.asarray(Xs))
    print(f"[{model}] PCA fit on {len(Xs)} rows; explained var={float(pca.explained_variance_ratio_.sum()):.3f}", flush=True)

    total = client.query(f"SELECT count() FROM mmcommons.emb_{model}").result_rows[0][0]
    if limit:
        total = min(total, limit)
    red = np.memmap(red_path, dtype=np.float32, mode="w+", shape=(total, pca_dim))
    fmd5 = open(md5_path, "w")
    off, last, t0 = 0, "", time.time()
    while off < total:
        md5, arr, last = fetch_batch(client, model, last, min(read_batch, total - off))
        if md5 is None:
            break
        red[off:off + len(md5)] = pca.transform(cp.asarray(arr))
        fmd5.write("\n".join(md5) + "\n")
        off += len(md5)
        if off % (read_batch * 20) == 0:
            print(f"[{model}] projected {off}/{total} ({off/(time.time()-t0):.0f}/s)", flush=True)
    red.flush(); fmd5.close()
    open(meta, "w").write(f"{off} {pca_dim}\n")
    try: os.rmdir(lock)                       # release build lock
    except OSError: pass
    print(f"[{model}] reduced done: N={off} in {time.time()-t0:.0f}s", flush=True)
    return red_path, md5_path, off


def insert_coords(client, table, md5_list, coords, has_z):
    import pyarrow as pa
    cols = {"md5": pa.array(md5_list, pa.string()),
            "x": pa.array(coords[:, 0], pa.float32()),
            "y": pa.array(coords[:, 1], pa.float32())}
    if has_z:
        cols["z"] = pa.array(coords[:, 2], pa.float32())
        cols["color"] = pa.array(coords[:, 3], pa.float32())
    else:
        cols["color"] = pa.array(coords[:, 2], pa.float32())
    client.insert_arrow(table, pa.table(cols))


def run_umap(client, model, comps, red_path, md5_path, N, pca_dim, args):
    import cupy as cp
    from cuml.manifold import UMAP
    table = f"mmcommons.umap_{model}_{'3d' if comps >= 4 else '2d'}"
    has_z = comps >= 4
    red = np.memmap(red_path, dtype=np.float32, mode="r", shape=(N, pca_dim))

    rng = np.random.default_rng(args.seed)
    k = min(args.fit_sample, N)
    idx = np.sort(rng.choice(N, size=k, replace=False))
    print(f"[{model}/{comps}c] fitting UMAP on {k} pts (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})", flush=True)
    um = UMAP(n_components=comps, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
              build_algo="nn_descent", random_state=args.seed, output_type="numpy")
    t0 = time.time()
    um.fit(cp.asarray(np.ascontiguousarray(red[idx])))
    print(f"[{model}/{comps}c] UMAP fit in {time.time()-t0:.0f}s; transforming {N} pts -> {table}", flush=True)

    # Truncate first so a restart re-does one clean pass (tables have no dedup; the reduce cache persists,
    # so only the transform repeats). Safe: this table is derived solely from this job.
    client.command(f"TRUNCATE TABLE {table}")
    md5f = open(md5_path)
    off, t1 = 0, time.time()
    while off < N:
        n = min(args.tf_batch, N - off)
        coords = np.asarray(um.transform(cp.asarray(np.ascontiguousarray(red[off:off + n]))))
        md5b = [md5f.readline().strip() for _ in range(n)]
        insert_coords(client, table, md5b, coords, has_z)
        off += n
        if off % (args.tf_batch * 10) == 0:
            print(f"[{model}/{comps}c] transformed {off}/{N} ({off/(time.time()-t1):.0f}/s)", flush=True)
    md5f.close()
    print(f"[{model}/{comps}c] DONE {off} pts in {time.time()-t1:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DIMS))
    ap.add_argument("--dims", default="2d,3d", help="which to build: 2d,3d")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-sample", type=int, default=3_000_000)
    ap.add_argument("--fit-sample", type=int, default=10_000_000)
    ap.add_argument("--read-batch", type=int, default=200_000)
    ap.add_argument("--tf-batch", type=int, default=500_000)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(__file__), "..", "umap_work"))
    ap.add_argument("--limit", type=int, default=0, help="cap N for a quick validation run (0=all)")
    ap.add_argument("--reduce-only", action="store_true", help="build the PCA-reduced cache and exit (no UMAP)")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    client = get_client()
    ensure_tables(client, args.model)
    red_path, md5_path, N = build_reduced(client, args.model, args.pca_dim,
                                          args.pca_sample, args.read_batch, args.work, args.limit)
    if args.reduce_only:
        print(f"[{args.model}] reduce-only done: N={N}", flush=True)
        return
    for d in [x.strip() for x in args.dims.split(",") if x.strip()]:
        run_umap(client, args.model, 4 if d == "3d" else 3, red_path, md5_path, N, args.pca_dim, args)
    print(f"[{args.model}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

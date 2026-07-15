"""Fidelity of the embedded Int8 quantizer vs full precision, and the 3 ways to use the code."""
import sys, numpy as np, pyarrow.parquet as pq, quant

X = np.stack(pq.read_table("qsample.parquet").column("v").to_numpy())  # (N,4096) float32
X = X.astype(np.float32)
N = X.shape[0]
nq = 2000
Q, DB = X[:nq], X[nq:]
print(f"loaded {N} vectors dim {X.shape[1]}; queries={nq} db={DB.shape[0]}")

def topk(sim, k=10):
    return np.argpartition(-sim, k, axis=1)[:, :k]

def norm(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-30)

# ground truth: full-precision cosine top-10
K = 10
gt = topk(norm(Q) @ norm(DB).T, K)

def recall(approx_top, gt_top):
    return np.mean([len(set(a) & set(g)) / K for a, g in zip(approx_top, gt_top)])

signs = quant.rotation_signs()
idxQ, idxDB = quant.quantize(Q, signs), quant.quantize(DB, signs)

print(f"\n{'method':<26}{'b=8':>8}{'b=4':>8}{'b=2':>8}{'b=1':>8}   (recall@10 vs full precision)")

# (A) correct: dequant via embedded Lloyd-Max codebook
row = []
for b in (8, 4, 2, 1):
    qr, dr = norm(quant.dequant(idxQ, b)), norm(quant.dequant(idxDB, b))
    row.append(recall(topk(qr @ dr.T), gt))
print(f"{'dequant (Lloyd-Max, correct)':<26}" + "".join(f"{r:>8.3f}" for r in row))

# (B) Q2: store fixed-point Lloyd-Max LEVEL in int8 (level*scale rounded), int cosine
row = []
for b in (8, 4, 2, 1):
    cb = quant.CODEBOOKS[b]
    scale = 127.0 / np.max(np.abs(cb))
    lvl = np.round(cb * scale)                       # int8 fixed-point levels
    qg, dg = (idxQ >> (8 - b)), (idxDB >> (8 - b))
    qr, dr = norm(lvl[qg].astype(np.float32)), norm(lvl[dg].astype(np.float32))
    row.append(recall(topk(qr @ dr.T), gt))
print(f"{'fixed-point level int8':<26}" + "".join(f"{r:>8.3f}" for r in row))

# (C) naive: use the raw cell INDEX directly as the vector (no codebook), int cosine
row = []
for b in (8, 4, 2, 1):
    qg, dg = (idxQ >> (8 - b)).astype(np.float32), (idxDB >> (8 - b)).astype(np.float32)
    c = (2 ** b - 1) / 2.0
    qr, dr = norm(qg - c), norm(dg - c)
    row.append(recall(topk(qr @ dr.T), gt))
print(f"{'raw index (no dequant)':<26}" + "".join(f"{r:>8.3f}" for r in row))

# similarity correlation (Pearson) on a sample of query-db pairs, dequant method
ts = (norm(Q) @ norm(DB).T)[:, :5000].ravel()
for b in (8, 4, 1):
    qr, dr = norm(quant.dequant(idxQ, b)), norm(quant.dequant(idxDB, b))
    asim = (qr @ dr.T)[:, :5000].ravel()
    print(f"Pearson(true, dequant b={b}) = {np.corrcoef(ts, asim)[0,1]:.4f}")

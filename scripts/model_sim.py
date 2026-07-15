import numpy as np, pyarrow.parquet as pq, glob, os, time
from scipy.stats import rankdata
np.random.seed(42)

RESULTS = sorted(glob.glob('results/*.parquet'))
N_RSA   = 5000      # sample for CKA/RSA (NxN matrices)
POOL    = 10000     # k-NN search pool (all)
K       = 10        # k for kNN overlap
T       = 500_000   # triplets

t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ---- load all models, aligned by id, L2-normalized rows ----
mats={}; ids_ref=None
for p in RESULTS:
    slug=os.path.basename(p)[:-8]
    t=pq.read_table(p, columns=['id','embedding'])
    ids=t.column('id').to_numpy()
    emb=t.column('embedding').combine_chunks()
    flat=np.asarray(emb.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    m=flat.reshape(len(ids), -1)
    o=np.argsort(ids); ids=ids[o]; m=m[o]
    if ids_ref is None: ids_ref=ids
    assert np.array_equal(ids, ids_ref), f"id mismatch {slug}"
    n=np.linalg.norm(m,axis=1,keepdims=True); n[n==0]=1.0
    mats[slug]=(m/n).astype(np.float32)
    log(f"loaded {slug:48s} shape={mats[slug].shape}")

models=list(mats.keys())
M=len(models)
# short labels by (dim, name) order already given by sorted glob? reorder by dim then name
order=sorted(range(M), key=lambda i:(mats[models[i]].shape[1], models[i]))
models=[models[i] for i in order]
labels=[f"m{i:02d}" for i in range(M)]
print("\n=== LEGEND ===");
for i,s in enumerate(models): print(f"  {labels[i]}  dim={mats[s].shape[1]:4d}  {s}")

# common sample indices
samp=np.random.choice(POOL, N_RSA, replace=False)

# ---- CKA (linear) : cosine of flattened double-centered Gram on sample ----
log("CKA: building centered grams")
G=np.empty((M, N_RSA*N_RSA), dtype=np.float32)
for i,s in enumerate(models):
    X=mats[s][samp]                      # N x d (already row-normalized)
    K_=X@X.T                             # N x N gram (cosine sims)
    K_-=K_.mean(0,keepdims=True); K_-=K_.mean(1,keepdims=True); K_+=K_.mean()  # double-center
    v=K_.ravel(); v/= (np.linalg.norm(v)+1e-12)
    G[i]=v
CKA=(G@G.T).astype(np.float64); del G
log("CKA done")

# ---- RSA : Spearman corr of pairwise cosine-distance RDM (upper triangle) ----
log("RSA: building ranked RDMs")
iu=np.triu_indices(N_RSA,1)
R=np.empty((M, iu[0].size), dtype=np.float32)
for i,s in enumerate(models):
    X=mats[s][samp]
    D=1.0-(X@X.T)                        # cosine distance
    r=rankdata(D[iu]).astype(np.float32) # spearman = pearson on ranks
    r-=r.mean(); r/=(np.linalg.norm(r)+1e-12)
    R[i]=r
RSA=(R@R.T).astype(np.float64); del R
log("RSA done")

# ---- kNN overlap : queries=sample, search full pool, k neighbours ----
log("kNN: computing neighbour sets")
knn={}
for s in models:
    X=mats[s]; Q=X[samp]
    sims=Q@X.T                           # Nq x POOL
    sims[np.arange(N_RSA), samp]=-2.0     # exclude self
    idx=np.argpartition(-sims, K, axis=1)[:,:K]
    knn[s]=np.sort(idx,axis=1)
KNN=np.eye(M)
for i in range(M):
    for j in range(i+1,M):
        a=knn[models[i]]; b=knn[models[j]]
        # overlap per query = |intersect|/K, averaged
        ov=0.0
        for qi in range(N_RSA):
            ov+=np.intersect1d(a[qi],b[qi],assume_unique=True).size
        v=ov/(N_RSA*K); KNN[i,j]=KNN[j,i]=v
log("kNN done")

# ---- Triplet agreement : sample triplets, do models agree which is closer ----
log("triplet: sampling")
A=np.random.randint(0,POOL,T); B=np.random.randint(0,POOL,T); C=np.random.randint(0,POOL,T)
bad=(A==B)|(A==C)|(B==C)
A,B,C=A[~bad],B[~bad],C[~bad]
closer=np.empty((M, A.size), dtype=bool)
for i,s in enumerate(models):
    X=mats[s]
    dab=np.einsum('ij,ij->i', X[A], X[B])   # cosine sim (higher=closer)
    dac=np.einsum('ij,ij->i', X[A], X[C])
    closer[i]= dab>dac
TRI=np.eye(M)
for i in range(M):
    for j in range(i+1,M):
        v=np.mean(closer[i]==closer[j]); TRI[i,j]=TRI[j,i]=v
log("triplet done")

def show(name, Mx, diag1=True):
    print(f"\n=== {name} (rows/cols = m00..m{M-1:02d}) ===")
    hdr="      "+" ".join(f"{labels[j][1:]:>4}" for j in range(M)); print(hdr)
    for i in range(M):
        row=" ".join(f"{Mx[i,j]:4.2f}" for j in range(M))
        print(f"{labels[i]}  {row}")
    np.savetxt(f"sim_{name}.csv", Mx, delimiter=",", fmt="%.4f")

show("CKA", CKA); show("RSA", RSA); show("kNN_overlap", KNN); show("Triplet_agreement", TRI)

# summary
def topbot(name,Mx):
    iu2=np.triu_indices(M,1); vals=Mx[iu2]
    order2=np.argsort(vals)
    print(f"\n-- {name}: most similar pairs --")
    for k in order2[::-1][:5]:
        i,j=iu2[0][k],iu2[1][k]; print(f"   {vals[k]:.3f}  {models[i]}  <>  {models[j]}")
    print(f"-- {name}: least similar pairs --")
    for k in order2[:5]:
        i,j=iu2[0][k],iu2[1][k]; print(f"   {vals[k]:.3f}  {models[i]}  <>  {models[j]}")
print("\n========= SUMMARY =========")
for nm,Mx in [("CKA",CKA),("RSA",RSA),("kNN_overlap",KNN),("Triplet_agreement",TRI)]:
    topbot(nm,Mx)
log("ALL DONE")

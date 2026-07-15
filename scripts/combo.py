import numpy as np, pyarrow.parquet as pq, time
from scipy.stats import rankdata
np.random.seed(0); t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}",flush=True)
MODELS={'gemini-2':('google_gemini_embedding_2','prefix'),
        'qwen3-8b':('qwen_qwen3_embedding_8b','prefix'),
        'all-mpnet-base':('sentence_transformers_all_mpnet_base_v2','pca')}
N_RSA=3000; POOL=10000; K=10; T=300_000
def load(slug):
    t=pq.read_table(f'results/{slug}.parquet',columns=['id','embedding']); ids=t.column('id').to_numpy()
    flat=np.asarray(t.column('embedding').combine_chunks().flatten().to_numpy(zero_copy_only=False),dtype=np.float32)
    m=flat.reshape(len(ids),-1)[np.argsort(ids)]; n=np.linalg.norm(m,axis=1,keepdims=True); n[n==0]=1
    return (m/n).astype(np.float32)
def unit(Y): n=np.linalg.norm(Y,axis=1,keepdims=True); n[n==0]=1; return (Y/n).astype(np.float32)
def fwht(a):
    a=a.copy(); P=a.shape[1]; h=1
    while h<P:
        a=a.reshape(a.shape[0],-1,2*h); x=a[:,:,:h].copy(); y=a[:,:,h:].copy()
        a[:,:,:h]=x+y; a[:,:,h:]=x-y; a=a.reshape(a.shape[0],P); h*=2
    return a
def lloyd_gauss(bits,iters=40):
    g=np.random.randn(300000); nl=2**bits; lv=np.quantile(g,(np.arange(nl)+0.5)/nl)
    for _ in range(iters):
        b=(lv[:-1]+lv[1:])/2; idx=np.searchsorted(b,g)
        for i in range(nl):
            s=g[idx==i]
            if s.size: lv[i]=s.mean()
    return np.sort(lv)
LL={b:lloyd_gauss(b) for b in [1,2,4,8]}
def had_lloyd(Y,bits):                      # Hadamard rotation + Gaussian Lloyd (the best low-bit quantizer)
    n=Y.shape[0]; D=Y.shape[1]; P=1<<int(np.ceil(np.log2(max(D,2))))
    rad=np.random.choice([-1.,1.],P).astype(np.float32)
    Yp=np.zeros((n,P),dtype=np.float32); Yp[:,:D]=unit(Y); Yr=fwht(Yp*rad)/np.sqrt(P)
    sd=Yr.std(0,keepdims=True); sd[sd==0]=1; lv=LL[bits]; bnd=(lv[:-1]+lv[1:])/2
    return (lv[np.searchsorted(bnd,Yr/sd)]*sd).astype(np.float32)
def metrics(Yn,ORIG):
    Yn=unit(Yn); samp,knn0,A,B,C,closer0=ORIG
    Q=Yn[samp]; sims=Q@Yn.T; sims[np.arange(len(samp)),samp]=-2
    idx=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    ov=sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(len(samp)))/(len(samp)*K)
    dab=np.einsum('ij,ij->i',Yn[A],Yn[B]); dac=np.einsum('ij,ij->i',Yn[A],Yn[C]); tri=float(np.mean((dab>dac)==closer0))
    return ov,tri
BITS=[16,8,4,2,1]   # 16 = full precision (no quant)
for name,(slug,drm) in MODELS.items():
    Xn=load(slug); D=Xn.shape[1]; log(f"==== {name} dim={D} DR={drm} ====")
    samp=np.random.choice(POOL,N_RSA,replace=False)
    sims=Xn[samp]@Xn.T; sims[np.arange(N_RSA),samp]=-2; knn0=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    A=np.random.randint(0,POOL,T);B=np.random.randint(0,POOL,T);C=np.random.randint(0,POOL,T)
    bad=(A==B)|(A==C)|(B==C);A,B,C=A[~bad],B[~bad],C[~bad]
    closer0=np.einsum('ij,ij->i',Xn[A],Xn[B])>np.einsum('ij,ij->i',Xn[A],Xn[C]); ORIG=(samp,knn0,A,B,C,closer0)
    mu=Xn.mean(0,keepdims=True); Xc=Xn-mu; Cm=(Xc.T@Xc).astype(np.float32); w,V=np.linalg.eigh(Cm); V=V[:,::-1]; PCp=Xc@V
    def reduce(k):
        if k>=D: return Xn
        return (PCp[:,:k] if drm=='pca' else Xn[:,:k])
    dims=[D,D//2,D//4,D//8,D//16]
    print(f"  {'dims':>6} {'bits':>4} {'totbits':>8} {'kNN':>6} {'Tri':>6}")
    rows=[]
    for k in dims:
        Yk=reduce(k)
        for b in BITS:
            Yq=Yk if b==16 else had_lloyd(Yk,b)
            ov,tri=metrics(Yq,ORIG); tb=k*b
            rows.append((k,b,tb,ov,tri)); print(f"  {k:6d} {b:4d} {tb:8d} {ov:6.3f} {tri:6.3f}",flush=True)
    # Pareto frontier (maximize kNN for <= total bits)
    rows.sort(key=lambda r:r[2])
    best=-1; print("  -- Pareto frontier (best kNN per bit budget) --")
    for k,b,tb,ov,tri in rows:
        if ov>best: best=ov; print(f"     totbits={tb:7d}  dims={k:5d} x {b:2d}bit   kNN={ov:.3f}")
    log(f"done {name}")
log("ALL DONE")

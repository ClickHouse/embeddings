import numpy as np, pyarrow.parquet as pq, time, os
from scipy.stats import rankdata
np.random.seed(0)
t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

MODELS = {  # short name -> parquet slug
 'gemini-2'      : 'google_gemini_embedding_2',
 'qwen3-8b'      : 'qwen_qwen3_embedding_8b',
 'all-mpnet-base': 'sentence_transformers_all_mpnet_base_v2',
}
N_RSA=3000; POOL=10000; K=10; T=300_000

def load(slug):
    t=pq.read_table(f'results/{slug}.parquet', columns=['id','embedding'])
    ids=t.column('id').to_numpy()
    flat=np.asarray(t.column('embedding').combine_chunks().flatten().to_numpy(zero_copy_only=False),dtype=np.float32)
    m=flat.reshape(len(ids),-1); o=np.argsort(ids); m=m[o]
    n=np.linalg.norm(m,axis=1,keepdims=True); n[n==0]=1
    return (m/n).astype(np.float32)

def unit(Y):
    n=np.linalg.norm(Y,axis=1,keepdims=True); n[n==0]=1
    return (Y/n).astype(np.float32)

# ---------- FWHT ----------
def fwht(a):  # a: (N, P) P power of two, transform along axis1
    a=a.copy(); P=a.shape[1]; h=1
    while h<P:
        a=a.reshape(a.shape[0], -1, 2*h)
        x=a[:,:,:h].copy(); y=a[:,:,h:].copy()
        a[:,:,:h]=x+y; a[:,:,h:]=x-y
        a=a.reshape(a.shape[0],P); h*=2
    return a

# ---------- BFloat16 top-k bit truncation (QBit) ----------
def qbit_bits(Xn,k):
    u=Xn.astype(np.float32).view(np.uint32)
    bf=(u>>16).astype(np.uint32)                  # top 16 bits = bfloat16
    mask=np.uint32((0xFFFF << (16-k)) & 0xFFFF)
    bf=bf & mask
    out=(bf<<16).astype(np.uint32)
    return out.view(np.float32)

# ---------- fp8 codebooks ----------
def fp8_codebook(e_bits,m_bits,bias,emax_excl_nan):
    vals=set([0.0])
    for e in range(0, 2**e_bits):
        if e==emax_excl_nan: continue  # reserve top exponent (nan/inf)
        for m in range(0,2**m_bits):
            if e==0: v=(m/2**m_bits)*2.0**(1-bias)          # subnormal
            else:    v=(1+m/2**m_bits)*2.0**(e-bias)        # normal
            vals.add(v); vals.add(-v)
    return np.array(sorted(vals),dtype=np.float64)

CB_E4M3=fp8_codebook(4,3,7,15)   # e4m3fn: top exponent reserved
CB_E5M2=fp8_codebook(5,2,15,31)  # e5m2: top exponent reserved

def quantize_codebook(x, cb):
    # nearest representable in sorted codebook cb
    idx=np.searchsorted(cb, x)
    idx=np.clip(idx,1,len(cb)-1)
    left=cb[idx-1]; right=cb[idx]
    return np.where(np.abs(x-left)<=np.abs(x-right), left, right)

def fp8_quant(Xn, cb, fp8max):
    scale=np.abs(Xn).max(0,keepdims=True); scale[scale==0]=1
    s=scale/fp8max
    q=quantize_codebook((Xn/s).astype(np.float64), cb)
    return (q*s).astype(np.float32)

# ---------- uniform int quant (per-dim symmetric absmax) ----------
def int_quant(Xn,bits):
    if bits==1: return np.sign(Xn).astype(np.float32)
    amax=np.abs(Xn).max(0,keepdims=True); amax[amax==0]=1
    qmax=2**(bits-1)-1
    s=amax/qmax
    return (np.clip(np.round(Xn/s),-qmax,qmax)*s).astype(np.float32)

# ---------- quantile (global codebook from sample) ----------
def quantile_codebooks(Xn):
    v=Xn[np.random.choice(Xn.shape[0],min(4000,Xn.shape[0]),replace=False)].ravel().astype(np.float64)
    v=np.sort(v); cbs={}
    for b in [8,7,6,5,4,2]:
        nl=2**b
        edges=np.quantile(v,np.linspace(0,1,nl+1))
        cents=np.array([v[(v>=edges[i])&(v<=edges[i+1])].mean() if np.any((v>=edges[i])&(v<=edges[i+1])) else (edges[i]+edges[i+1])/2 for i in range(nl)])
        cbs[b]=np.sort(np.unique(cents))
    return cbs
def quantile_quant(Xn,cb):
    return quantize_codebook(Xn.astype(np.float64),cb).astype(np.float32)

# ---------- Lloyd-Max gaussian levels ----------
def lloyd_gauss(bits,iters=40):
    g=np.random.randn(400000); nl=2**bits
    lv=np.quantile(g,(np.arange(nl)+0.5)/nl)
    for _ in range(iters):
        b=(lv[:-1]+lv[1:])/2; idx=np.searchsorted(b,g)
        for i in range(nl):
            sel=g[idx==i]
            if sel.size: lv[i]=sel.mean()
    return np.sort(lv)
LLOYD={b:lloyd_gauss(b) for b in [1,2,3,4]}

# ---------- random projections (cosine is scale-invariant, so no scaling needed) ----------
def proj_gaussian(Xn,d): return Xn@np.random.randn(Xn.shape[1],d).astype(np.float32)
def proj_sparse_pm1(Xn,d): return Xn@np.random.choice([-1.,1.],(Xn.shape[1],d)).astype(np.float32)
def proj_sparse_106(Xn,d):
    R=np.random.choice([-1.,0.,1.],(Xn.shape[1],d),p=[1/6,2/3,1/6]).astype(np.float32); return Xn@R
def proj_very_sparse(Xn,d):
    D=Xn.shape[1]; s=max(1.0,np.sqrt(D)); p=1/s
    R=np.random.choice([-1.,0.,1.],(D,d),p=[p/2,1-p,p/2]).astype(np.float32); return Xn@R

def metrics(Yn, ORIG):
    Yn=unit(Yn)
    samp,knn0,rdm0,cka0,A,B,C,closer0=ORIG
    # kNN
    Q=Yn[samp]; sims=Q@Yn.T; sims[np.arange(len(samp)),samp]=-2
    idx=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    ov=sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(len(samp)))/(len(samp)*K)
    # RSA
    S=Yn[samp]; D=1-(S@S.T); iu=np.triu_indices(len(samp),1)
    r=rankdata(D[iu]).astype(np.float32); r-=r.mean(); r/=(np.linalg.norm(r)+1e-12)
    rsa=float(r@rdm0)
    # CKA
    Kg=S@S.T; Kg-=Kg.mean(0,keepdims=True); Kg-=Kg.mean(1,keepdims=True); Kg+=Kg.mean()
    v=Kg.ravel(); v/=(np.linalg.norm(v)+1e-12); cka=float(v@cka0)
    # triplet
    dab=np.einsum('ij,ij->i',Yn[A],Yn[B]); dac=np.einsum('ij,ij->i',Yn[A],Yn[C])
    tri=float(np.mean((dab>dac)==closer0))
    return ov,rsa,cka,tri

for name,slug in MODELS.items():
    Xn=load(slug); D=Xn.shape[1]
    log(f"==== {name}  dim={D} ====")
    samp=np.random.choice(POOL,N_RSA,replace=False)
    # originals
    sims=Xn[samp]@Xn.T; sims[np.arange(N_RSA),samp]=-2
    knn0=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    S=Xn[samp]; Dm=1-(S@S.T); iu=np.triu_indices(N_RSA,1)
    rdm0=rankdata(Dm[iu]).astype(np.float32); rdm0-=rdm0.mean(); rdm0/=np.linalg.norm(rdm0)
    Kg=S@S.T; Kg-=Kg.mean(0,keepdims=True); Kg-=Kg.mean(1,keepdims=True); Kg+=Kg.mean()
    cka0=Kg.ravel(); cka0/=np.linalg.norm(cka0)
    A=np.random.randint(0,POOL,T); B=np.random.randint(0,POOL,T); C=np.random.randint(0,POOL,T)
    bad=(A==B)|(A==C)|(B==C); A,B,C=A[~bad],B[~bad],C[~bad]
    dab=np.einsum('ij,ij->i',Xn[A],Xn[B]); dac=np.einsum('ij,ij->i',Xn[A],Xn[C]); closer0=dab>dac
    ORIG=(samp,knn0,rdm0,cka0,A,B,C,closer0)

    # precompute orthogonal Q (Haar) for orthogonal proj + turboquant rotation
    G=np.random.randn(D,D).astype(np.float32); Qo,Rr=np.linalg.qr(G); Qo*=np.sign(np.diag(Rr))[None,:]
    # SRHT precompute
    P=1<<int(np.ceil(np.log2(D))); rad=np.random.choice([-1.,1.],P).astype(np.float32)
    Xpad=np.zeros((Xn.shape[0],P),dtype=np.float32); Xpad[:,:D]=Xn*rad[:D]
    HX=fwht(Xpad); perm=np.random.permutation(P)

    def emit(method,param,Yn):
        ov,rsa,cka,tri=metrics(Yn,ORIG)
        print(f"  {method:14s} {str(param):>10s} | kNN={ov:.3f} RSA={rsa:.3f} CKA={cka:.3f} Tri={tri:.3f}",flush=True)

    print(" --- QBit (BFloat16 top-k bits) ---")
    for k in range(1,16): emit('qbit_bits',k, qbit_bits(Xn,k))

    print(" --- dim quantization ---")
    emit('fp8_e4m3','-', fp8_quant(Xn,CB_E4M3,448.0))
    emit('fp8_e5m2','-', fp8_quant(Xn,CB_E5M2,57344.0))
    for b in [8,4,2,1]:
        nm={8:'int8',4:'int4',2:'int2',1:'binary'}[b]; emit('uniform',nm, int_quant(Xn,b))

    print(" --- quantile quantization ---")
    qcb=quantile_codebooks(Xn)
    for b in [8,7,6,5,4,2]: emit('quantile',f'{b}bit', quantile_quant(Xn,qcb[b]))

    print(" --- TurboQuant (rand-rotation + Gaussian Lloyd-Max) ---")
    Xr=Xn@Qo                                  # rotate (orthogonal => preserves inner products)
    sd=Xr.std(0,keepdims=True); sd[sd==0]=1
    for b in [1,2,3,4]:
        lv=LLOYD[b]; bnd=(lv[:-1]+lv[1:])/2
        idx=np.searchsorted(bnd,(Xr/sd)); Yr=lv[idx]*sd
        emit('turboquant',f'{b}bit', Yr.astype(np.float32))
    # turboquant WITHOUT rotation (ablation), 2-bit
    sd2=Xn.std(0,keepdims=True); sd2[sd2==0]=1
    lv=LLOYD[2]; bnd=(lv[:-1]+lv[1:])/2; idx=np.searchsorted(bnd,(Xn/sd2)); emit('turbo_norot','2bit',(lv[idx]*sd2).astype(np.float32))

    print(" --- random projections (dim ladder x type) ---")
    dims=[]; d=D
    while d>=1:
        d//=2
        if d>=1: dims.append(d)
    if dims[-1]!=1: dims.append(1)
    for d in dims:
        emit('proj_gauss',  d, proj_gaussian(Xn,d))
        emit('proj_ortho',  d, Xn@Qo[:,:d])
        emit('proj_pm1',    d, proj_sparse_pm1(Xn,d))
        emit('proj_106',    d, proj_sparse_106(Xn,d))
        emit('proj_vsparse',d, proj_very_sparse(Xn,d))
        emit('proj_srht',   d, HX[:,perm[:d]])
    log(f"done {name}")
log("ALL DONE")

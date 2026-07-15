import numpy as np, pyarrow.parquet as pq, time
from scipy.stats import rankdata
np.random.seed(0); t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}",flush=True)
MODELS={'gemini-2':'google_gemini_embedding_2','qwen3-8b':'qwen_qwen3_embedding_8b','all-mpnet-base':'sentence_transformers_all_mpnet_base_v2'}
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
def int_quant(Xn,bits):
    if bits==1: return np.sign(Xn).astype(np.float32)
    amax=np.abs(Xn).max(0,keepdims=True); amax[amax==0]=1; qmax=2**(bits-1)-1; s=amax/qmax
    return (np.clip(np.round(Xn/s),-qmax,qmax)*s).astype(np.float32)
def lloyd_gauss(bits,iters=40):
    g=np.random.randn(400000); nl=2**bits; lv=np.quantile(g,(np.arange(nl)+0.5)/nl)
    for _ in range(iters):
        b=(lv[:-1]+lv[1:])/2; idx=np.searchsorted(b,g)
        for i in range(nl):
            sel=g[idx==i];
            if sel.size: lv[i]=sel.mean()
    return np.sort(lv)
LL={b:lloyd_gauss(b) for b in [1,2,4]}
def lloyd_quant(X,bits):
    sd=X.std(0,keepdims=True); sd[sd==0]=1; lv=LL[bits]; bnd=(lv[:-1]+lv[1:])/2
    return (lv[np.searchsorted(bnd,X/sd)]*sd).astype(np.float32)
def sphere_quant(X,bits,dim):
    # FULLY data-oblivious: hard-coded Gaussian/sphere-optimal levels x (1/sqrt(dim))
    lv=LL[bits]; bnd=(lv[:-1]+lv[1:])/2; s=1.0/np.sqrt(dim)
    return (lv[np.searchsorted(bnd,X/s)]*s).astype(np.float32)

def metrics(Yn,ORIG):
    Yn=unit(Yn); samp,knn0,rdm0,cka0,A,B,C,closer0=ORIG
    Q=Yn[samp]; sims=Q@Yn.T; sims[np.arange(len(samp)),samp]=-2
    idx=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    ov=sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(len(samp)))/(len(samp)*K)
    S=Yn[samp]; D=1-(S@S.T); iu=np.triu_indices(len(samp),1)
    r=rankdata(D[iu]).astype(np.float32); r-=r.mean(); r/=(np.linalg.norm(r)+1e-12); rsa=float(r@rdm0)
    Kg=S@S.T; Kg-=Kg.mean(0,keepdims=True); Kg-=Kg.mean(1,keepdims=True); Kg+=Kg.mean()
    v=Kg.ravel(); v/=(np.linalg.norm(v)+1e-12); cka=float(v@cka0)
    dab=np.einsum('ij,ij->i',Yn[A],Yn[B]); dac=np.einsum('ij,ij->i',Yn[A],Yn[C]); tri=float(np.mean((dab>dac)==closer0))
    return ov,rsa,cka,tri

for name,slug in MODELS.items():
    Xn=load(slug); D=Xn.shape[1]; P=1<<int(np.ceil(np.log2(D)))
    log(f"==== {name} dim={D} (Hadamard pad P={P}) ====")
    samp=np.random.choice(POOL,N_RSA,replace=False)
    sims=Xn[samp]@Xn.T; sims[np.arange(N_RSA),samp]=-2; knn0=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    S=Xn[samp]; Dm=1-(S@S.T); iu=np.triu_indices(N_RSA,1)
    rdm0=rankdata(Dm[iu]).astype(np.float32); rdm0-=rdm0.mean(); rdm0/=np.linalg.norm(rdm0)
    Kg=S@S.T; Kg-=Kg.mean(0,keepdims=True); Kg-=Kg.mean(1,keepdims=True); Kg+=Kg.mean(); cka0=Kg.ravel(); cka0/=np.linalg.norm(cka0)
    A=np.random.randint(0,POOL,T);B=np.random.randint(0,POOL,T);C=np.random.randint(0,POOL,T)
    bad=(A==B)|(A==C)|(B==C);A,B,C=A[~bad],B[~bad],C[~bad]
    dab=np.einsum('ij,ij->i',Xn[A],Xn[B]); dac=np.einsum('ij,ij->i',Xn[A],Xn[C]); closer0=dab>dac
    ORIG=(samp,knn0,rdm0,cka0,A,B,C,closer0)
    # Randomized Hadamard rotation (orthonormal): y = FWHT(rad * pad(x))/sqrt(P)
    rad=np.random.choice([-1.,1.],P).astype(np.float32)
    Xp=np.zeros((Xn.shape[0],P),dtype=np.float32); Xp[:,:D]=Xn; Xr=(fwht(Xp*rad)/np.sqrt(P)).astype(np.float32)
    def emit(q,rot,Yn): ov,rsa,cka,tri=metrics(Yn,ORIG); print(f"  {q:10s} {'+RHT' if rot else 'plain':6s} | kNN={ov:.3f} RSA={rsa:.3f} CKA={cka:.3f} Tri={tri:.3f}",flush=True)
    for q,fn in [('binary',lambda Z:int_quant(Z,1)),('int2',lambda Z:int_quant(Z,2)),
                 ('lloyd1',lambda Z:lloyd_quant(Z,1)),('lloyd2',lambda Z:lloyd_quant(Z,2)),('lloyd4',lambda Z:lloyd_quant(Z,4))]:
        emit(q,False,fn(Xn)); emit(q,True, fn(Xr))
    # fully data-oblivious hard-coded sphere levels (no calibration), bits 1/2/4
    for b in [1,2,4]:
        emit(f'sphere{b}',False, sphere_quant(Xn,b,D))
        emit(f'sphere{b}',True,  sphere_quant(Xr,b,P))
    log(f"done {name}")
log("ALL DONE")

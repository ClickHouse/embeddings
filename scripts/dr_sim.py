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
    Xn=load(slug); D=Xn.shape[1]; log(f"==== {name} dim={D} ====")
    samp=np.random.choice(POOL,N_RSA,replace=False)
    sims=Xn[samp]@Xn.T; sims[np.arange(N_RSA),samp]=-2; knn0=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    S=Xn[samp]; Dm=1-(S@S.T); iu=np.triu_indices(N_RSA,1)
    rdm0=rankdata(Dm[iu]).astype(np.float32); rdm0-=rdm0.mean(); rdm0/=np.linalg.norm(rdm0)
    Kg=S@S.T; Kg-=Kg.mean(0,keepdims=True); Kg-=Kg.mean(1,keepdims=True); Kg+=Kg.mean(); cka0=Kg.ravel(); cka0/=np.linalg.norm(cka0)
    A=np.random.randint(0,POOL,T);B=np.random.randint(0,POOL,T);C=np.random.randint(0,POOL,T)
    bad=(A==B)|(A==C)|(B==C);A,B,C=A[~bad],B[~bad],C[~bad]
    closer0=np.einsum('ij,ij->i',Xn[A],Xn[B])>np.einsum('ij,ij->i',Xn[A],Xn[C])
    ORIG=(samp,knn0,rdm0,cka0,A,B,C,closer0)
    # PCA basis (covariance eigendecomp, descending)
    mu=Xn.mean(0,keepdims=True); Xc=Xn-mu
    Cm=(Xc.T@Xc).astype(np.float32); w,V=np.linalg.eigh(Cm); V=V[:,::-1]
    PCp=Xc@V    # full PCA scores, take prefix per k
    # random orthogonal basis
    G=np.random.randn(D,D).astype(np.float32); Qo,Rr=np.linalg.qr(G); Qo*=np.sign(np.diag(Rr))[None,:]; RPo=Xn@Qo
    # SRHT
    P=1<<int(np.ceil(np.log2(D))); rad=np.random.choice([-1.,1.],P).astype(np.float32)
    Xp=np.zeros((Xn.shape[0],P),dtype=np.float32); Xp[:,:D]=Xn*rad[:D]; HX=fwht(Xp); perm=np.random.permutation(P)
    dims=[]; k=D
    while k>1:
        k//=2; dims.append(max(k,1))
    if dims[-1]!=1: dims.append(1)
    for k in dims:
        ratio=f"1/{D//k}" if k>0 else "?"
        Gk=np.random.randn(D,k).astype(np.float32)            # rectangular Gaussian
        Uk=np.random.uniform(-1,1,(D,k)).astype(np.float32)   # rectangular Uniform
        for meth,Y in [('pca',PCp[:,:k]),('prefix',Xn[:,:k]),('rand_gauss',Xn@Gk),('rand_unif',Xn@Uk),
                       ('rand_ortho',RPo[:,:k]),('srht_hadamard',HX[:,perm[:k]])]:
            ov,rsa,cka,tri=metrics(Y,ORIG)
            print(f"  d={k:5d} (~{ratio:>6s}) {meth:10s} | kNN={ov:.3f} RSA={rsa:.3f} CKA={cka:.3f} Tri={tri:.3f}",flush=True)
    log(f"done {name}")
log("ALL DONE")

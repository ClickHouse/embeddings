import numpy as np, pyarrow.parquet as pq, time
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
def lloyd_gauss(bits,it=40):
    g=np.random.randn(300000); nl=2**bits; lv=np.quantile(g,(np.arange(nl)+.5)/nl)
    for _ in range(it):
        b=(lv[:-1]+lv[1:])/2; idx=np.searchsorted(b,g)
        for i in range(nl):
            s=g[idx==i]
            if s.size: lv[i]=s.mean()
    return np.sort(lv)
LL={1:lloyd_gauss(1),2:lloyd_gauss(2),4:lloyd_gauss(4)}
def metrics(Yn,ORIG):
    Yn=unit(Yn); samp,knn0,A,B,C,closer0=ORIG
    sims=Yn[samp]@Yn.T; sims[np.arange(len(samp)),samp]=-2
    idx=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    ov=sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(len(samp)))/(len(samp)*K)
    dab=np.einsum('ij,ij->i',Yn[A],Yn[B]); dac=np.einsum('ij,ij->i',Yn[A],Yn[C]); tri=float(np.mean((dab>dac)==closer0))
    return ov,tri
for name,slug in MODELS.items():
    Xn=load(slug); D=Xn.shape[1]; log(f"==== {name} dim={D} ====")
    samp=np.random.choice(POOL,N_RSA,replace=False)
    sims=Xn[samp]@Xn.T; sims[np.arange(N_RSA),samp]=-2; knn0=np.sort(np.argpartition(-sims,K,axis=1)[:,:K],axis=1)
    A=np.random.randint(0,POOL,T);B=np.random.randint(0,POOL,T);C=np.random.randint(0,POOL,T)
    bad=(A==B)|(A==C)|(B==C);A,B,C=A[~bad],B[~bad],C[~bad]
    closer0=np.einsum('ij,ij->i',Xn[A],Xn[B])>np.einsum('ij,ij->i',Xn[A],Xn[C]); ORIG=(samp,knn0,A,B,C,closer0)
    print("  -- EXPAND via random projection then 1-bit sign (SimHash/LSH) --")
    for mult in [0.25,0.5,1,2,4,8,16]:
        m=max(1,int(D*mult)); R=np.random.randn(D,m).astype(np.float32)
        Bsign=np.sign(Xn@R).astype(np.float32)
        ov,tri=metrics(Bsign,ORIG); print(f"     m={m:6d} (x{mult:<4}) 1bit  totbits={m:7d}  kNN={ov:.3f} Tri={tri:.3f}",flush=True)
    print("  -- reference: native D dims at b bits (Hadamard+Lloyd) --")
    P=1<<int(np.ceil(np.log2(D))); rad=np.random.choice([-1.,1.],P).astype(np.float32)
    Xp=np.zeros((Xn.shape[0],P),dtype=np.float32); Xp[:,:D]=Xn; Xr=np.empty_like(Xp)
    # FWHT
    a=(Xp*rad).copy(); h=1
    while h<P:
        a=a.reshape(a.shape[0],-1,2*h); x=a[:,:,:h].copy(); y=a[:,:,h:].copy(); a[:,:,:h]=x+y; a[:,:,h:]=x-y; a=a.reshape(a.shape[0],P); h*=2
    Xr=a/np.sqrt(P); sd=Xr.std(0,keepdims=True); sd[sd==0]=1
    for b in [1,2,4]:
        lv=LL[b]; bnd=(lv[:-1]+lv[1:])/2; Yq=lv[np.searchsorted(bnd,Xr/sd)]*sd
        ov,tri=metrics(Yq.astype(np.float32),ORIG); print(f"     native D={D} {b}bit  totbits={D*b:7d}  kNN={ov:.3f} Tri={tri:.3f}",flush=True)
    log(f"done {name}")
log("ALL DONE")

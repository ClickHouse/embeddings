import numpy as np, pyarrow.parquet as pq
np.random.seed(2)
M={'gemini-2':'google_gemini_embedding_2','qwen3-8b':'qwen_qwen3_embedding_8b','all-mpnet-base':'sentence_transformers_all_mpnet_base_v2'}
POOL=10000;K=10;N=3000
def load(s):
    t=pq.read_table(f'results/{s}.parquet',columns=['id','embedding']);ids=t.column('id').to_numpy()
    f=np.asarray(t.column('embedding').combine_chunks().flatten().to_numpy(zero_copy_only=False),dtype=np.float32)
    m=f.reshape(len(ids),-1)[np.argsort(ids)];n=np.linalg.norm(m,axis=1,keepdims=True);n[n==0]=1;return (m/n).astype(np.float32)
def unit(Y):n=np.linalg.norm(Y,axis=1,keepdims=True);n[n==0]=1;return (Y/n).astype(np.float32)
def lloyd(bits,it=40):
    g=np.random.randn(300000);nl=2**bits;lv=np.quantile(g,(np.arange(nl)+.5)/nl)
    for _ in range(it):
        b=(lv[:-1]+lv[1:])/2;idx=np.searchsorted(b,g)
        for i in range(nl):
            s=g[idx==i]
            if s.size:lv[i]=s.mean()
    return np.sort(lv)
L2,L4=lloyd(2),lloyd(4)
def q(Y,lv):
    sd=Y.std(0,keepdims=True);sd[sd==0]=1;bnd=(lv[:-1]+lv[1:])/2;return (lv[np.searchsorted(bnd,Y/sd)]*sd).astype(np.float32)
def fwht(a):
    a=a.copy();P=a.shape[1];h=1
    while h<P:
        a=a.reshape(a.shape[0],-1,2*h);x=a[:,:,:h].copy();y=a[:,:,h:].copy();a[:,:,:h]=x+y;a[:,:,h:]=x-y;a=a.reshape(a.shape[0],P);h*=2
    return a
def knn(Yn,samp,knn0):
    Yn=unit(Yn);s=Yn[samp]@Yn.T;s[np.arange(N),samp]=-2;idx=np.sort(np.argpartition(-s,K,1)[:,:K],1)
    return sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(N))/(N*K)
def rot(D):
    G=np.random.randn(D,D).astype(np.float32);Q,R=np.linalg.qr(G);return (Q*np.sign(np.diag(R))[None,:]).astype(np.float32)
for nm,sl in M.items():
    Xn=load(sl);D=Xn.shape[1];samp=np.random.choice(POOL,N,replace=False)
    s=Xn[samp]@Xn.T;s[np.arange(N),samp]=-2;knn0=np.sort(np.argpartition(-s,K,1)[:,:K],1)
    print(f"== {nm} D={D}  (all rows = 4D = {4*D} bits/vector) ==")
    # native: Hadamard rotate then Lloyd
    P=1<<int(np.ceil(np.log2(D)));rad=np.random.choice([-1.,1.],P).astype(np.float32)
    Xp=np.zeros((Xn.shape[0],P),dtype=np.float32);Xp[:,:D]=Xn;Xr=fwht(Xp*rad)/np.sqrt(P)
    print(f"  native  D x4bit            kNN={knn(q(Xr,L4),samp,knn0):.3f}   (4D bits)")
    print(f"  [ref] native D x2bit       kNN={knn(q(Xr,L2),samp,knn0):.3f}   (2D bits)")
    # expand 2x via Gaussian, then 2bit
    Rg=np.random.randn(D,2*D).astype(np.float32);Yg=Xn@Rg
    print(f"  expand2x Gaussian  x2bit   kNN={knn(q(Yg,L2),samp,knn0):.3f}   (4D bits)")
    # expand 2x via tight frame (two stacked random rotations), then 2bit
    Yf=np.hstack([Xn@rot(D),Xn@rot(D)])
    print(f"  expand2x tightframe x2bit  kNN={knn(q(Yf,L2),samp,knn0):.3f}   (4D bits)")
    # expand 2x via SRHT-to-2D then 2bit
    P2=1<<int(np.ceil(np.log2(2*D)));rad2=np.random.choice([-1.,1.],P2).astype(np.float32)
    Xp2=np.zeros((Xn.shape[0],P2),dtype=np.float32);Xp2[:,:D]=Xn;HX=fwht(Xp2*rad2[:P2]);perm=np.random.permutation(P2)
    print(f"  expand2x SRHT      x2bit   kNN={knn(q(HX[:,perm[:2*D]],L2),samp,knn0):.3f}   (4D bits)")

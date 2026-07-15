import numpy as np, pyarrow.parquet as pq
np.random.seed(0); POOL=10000;K=10;N=3000
def load(s):
    t=pq.read_table(f'results/{s}.parquet',columns=['id','embedding']);ids=t.column('id').to_numpy()
    f=np.asarray(t.column('embedding').combine_chunks().flatten().to_numpy(zero_copy_only=False),dtype=np.float32)
    m=f.reshape(len(ids),-1)[np.argsort(ids)];n=np.linalg.norm(m,axis=1,keepdims=True);n[n==0]=1;return (m/n).astype(np.float32)
def unit(Y):n=np.linalg.norm(Y,axis=1,keepdims=True);n[n==0]=1;return (Y/n).astype(np.float32)
def knn(Yn,samp,knn0):
    Yn=unit(Yn);s=Yn[samp]@Yn.T;s[np.arange(N),samp]=-2;idx=np.sort(np.argpartition(-s,K,1)[:,:K],1)
    return sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(N))/(N*K)
for nm,sl in [('3-small','openai_text_embedding_3_small'),('3-large','openai_text_embedding_3_large'),('ada-002','openai_text_embedding_ada_002')]:
    X=load(sl);D=X.shape[1];samp=np.random.choice(POOL,N,replace=False)
    s=X[samp]@X.T;s[np.arange(N),samp]=-2;knn0=np.sort(np.argpartition(-s,K,1)[:,:K],1)
    res=[]
    for k in [D,1024,512,256,128,64]:
        if k>D: continue
        res.append(f"d={k}:{knn(X[:,:k],samp,knn0):.3f}")
    print(f"{nm:8s} (D={D}) prefix-truncation kNN@10 self-fidelity:  "+"  ".join(res))

import numpy as np, pyarrow.parquet as pq
np.random.seed(1)
M={'qwen3-8b':'qwen_qwen3_embedding_8b','all-mpnet-base':'sentence_transformers_all_mpnet_base_v2'}
POOL=10000;K=10;N=3000
def load(s):
    t=pq.read_table(f'results/{s}.parquet',columns=['id','embedding']);ids=t.column('id').to_numpy()
    f=np.asarray(t.column('embedding').combine_chunks().flatten().to_numpy(zero_copy_only=False),dtype=np.float32)
    m=f.reshape(len(ids),-1)[np.argsort(ids)];n=np.linalg.norm(m,axis=1,keepdims=True);n[n==0]=1;return (m/n).astype(np.float32)
def unit(Y):n=np.linalg.norm(Y,axis=1,keepdims=True);n[n==0]=1;return (Y/n).astype(np.float32)
def knn(Yn,samp,knn0):
    Yn=unit(Yn);s=Yn[samp]@Yn.T;s[np.arange(N),samp]=-2
    idx=np.sort(np.argpartition(-s,K,1)[:,:K],1)
    return sum(np.intersect1d(idx[i],knn0[i],assume_unique=True).size for i in range(N))/(N*K)
for nm,sl in M.items():
    Xn=load(sl);D=Xn.shape[1];samp=np.random.choice(POOL,N,replace=False)
    s=Xn[samp]@Xn.T;s[np.arange(N),samp]=-2;knn0=np.sort(np.argpartition(-s,K,1)[:,:K],1)
    print(f"== {nm} D={D} ==")
    for mult in [0.5,1,2,4]:
        m=int(D*mult);R=np.random.randn(D,m).astype(np.float32);B=np.sign(Xn@R).astype(np.float32)
        print(f"  expand m={m} (x{mult}) 1bit  totbits={m}  kNN={knn(B,samp,knn0):.3f}")

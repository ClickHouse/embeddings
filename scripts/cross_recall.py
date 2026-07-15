#!/usr/bin/env python3
"""How differently do gemini-embedding-2 and nemotron-vl organize THIS dataset?
Use each photo's 1024px(l) embedding (common to both), and measure neighbor agreement:
for each photo, do the two models' top-K nearest photos overlap?"""
import glob, numpy as np

def load_size(tag, size='l'):
    d={}
    for m in glob.glob(f"emb/{tag}/chunk_*.meta.tsv"):
        mat=np.load(m.replace(".meta.tsv",".npy"))
        for i,l in enumerate(open(m).read().splitlines()[1:]):
            a=l.split("\t")
            if a[1]==size and a[3]=="1":
                v=mat[i]; n=np.linalg.norm(v)
                if n>0: d[a[0]]=v/n
    return d

for SIZE in ['l','o']:
    G=load_size("gemini_embedding_2",SIZE); N=load_size("nemotron_vl",SIZE)
    common=sorted(set(G)&set(N))
    if len(common)<50: continue
    Gm=np.stack([G[p] for p in common]); Nm=np.stack([N[p] for p in common])
    Gs=Gm@Gm.T; Ns=Nm@Nm.T
    np.fill_diagonal(Gs,-2); np.fill_diagonal(Ns,-2)
    n=len(common)
    print(f"\n=== neighbor agreement at size '{SIZE}'  (n={n} photos, random@K≈K/{n-1}) ===")
    print(f"  {'K':>4} {'overlap@K':>10} {'mutual NN match':>16}")
    for K in [1,5,10,20,50]:
        if K>=n: break
        Gtop=np.argpartition(-Gs,K,axis=1)[:,:K]
        Ntop=np.argpartition(-Ns,K,axis=1)[:,:K]
        ov=np.mean([len(set(Gtop[i])&set(Ntop[i]))/K for i in range(n)])
        nn1=np.mean([ (sorted(Gtop[i],key=lambda j:-Gs[i,j])[0]==sorted(Ntop[i],key=lambda j:-Ns[i,j])[0]) for i in range(n)]) if K>=1 else 0
        line=f"  {K:>4} {ov:>10.3f}"
        if K==1: line+=f" {nn1:>16.3f}"
        print(line)
    # global ranking correlation: avg Spearman of per-query distance orderings
    import numpy as _np
    def rankcorr(a,b):
        ra=_np.argsort(_np.argsort(a)); rb=_np.argsort(_np.argsort(b))
        return _np.corrcoef(ra,rb)[0,1]
    sp=_np.mean([rankcorr(Gs[i],Ns[i]) for i in range(0,n,5)])
    print(f"  mean per-query rank correlation (Spearman) of full neighbor ordering: {sp:.3f}")

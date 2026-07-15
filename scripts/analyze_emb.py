#!/usr/bin/env python3
"""How do image embeddings change with image size? Per model:
 A) cosine similarity of each size's embedding to the SAME photo's original (o) and to its 1024px (l)
 B) cross-size retrieval: query with size s, catalog = all photos' 1024px (l) embedding -> recall@1/@5
"""
import sys, glob, numpy as np
ORDER=['sq','q','t','s','n','w','m','z','c','l','h','k','3k','4k','5k','6k','o']
PX={'sq':75,'q':150,'t':100,'s':240,'n':320,'w':400,'m':500,'z':640,'c':800,'l':1024,
    'h':1600,'k':2048,'3k':3072,'4k':4096,'5k':5120,'6k':6144,'o':99999}
TAGS={"nemotron_vl":"nvidia llama-nemotron-embed-vl-1b-v2 (2048d, free)",
      "gemini_embedding_2":"google gemini-embedding-2 (3072d)",
      "gemini_embedding_2_preview":"google gemini-embedding-2-preview (3072d)"}

def load(tag):
    by={}
    for meta in sorted(glob.glob(f"emb/{tag}/chunk_*.meta.tsv"),
                       key=lambda p:int(p.split('chunk_')[1].split('.')[0])):
        mat=np.load(meta.replace(".meta.tsv",".npy"))
        rows=[l.rstrip("\n").split("\t") for l in open(meta)][1:]
        for i,r in enumerate(rows):
            if r[3]!="1": continue
            v=mat[i]; n=np.linalg.norm(v)
            if n==0: continue
            by.setdefault(r[0],{})[r[1]]=v/n          # store unit-normalized
    return by

def cos(a,b): return float(np.dot(a,b))   # already unit norm

def analyze(tag):
    by=load(tag); dim=len(next(iter(next(iter(by.values())).values())))
    print(f"\n{'='*78}\n{TAGS.get(tag,tag)}\n  photos={len(by)}  dim={dim}")
    # ---- A) drift vs original (o) and vs 1024 (l) ----
    for ref in ("o","l"):
        print(f"\n  A) cosine similarity to same photo's '{ref}' ({PX[ref] if ref!='o' else 'orig'}px):")
        print(f"     {'size':>4} {'px':>6} {'n':>5} {'mean_cos':>9} {'median':>8} {'p10':>7}")
        for s in ORDER:
            if s==ref: continue
            vals=[cos(d[s],d[ref]) for d in by.values() if s in d and ref in d]
            if len(vals)<5: continue
            a=np.array(vals)
            print(f"     {s:>4} {PX[s]:>6} {len(a):>5} {a.mean():>9.4f} {np.median(a):>8.4f} {np.percentile(a,10):>7.4f}")
    # ---- B) cross-size retrieval against 1024px catalog ----
    cat_ids=[pid for pid,d in by.items() if 'l' in d]
    if len(cat_ids)>=50:
        C=np.stack([by[p]['l'] for p in cat_ids])             # catalog matrix (unit rows)
        pos={p:i for i,p in enumerate(cat_ids)}
        print(f"\n  B) retrieval: query=size s, catalog={len(cat_ids)} photos' 1024px(l). recall@1 / @5 (self-match):")
        print(f"     {'size':>4} {'n_q':>5} {'r@1':>7} {'r@5':>7}")
        for s in ORDER:
            qids=[p for p in cat_ids if s in by[p]]
            if len(qids)<20: continue
            Q=np.stack([by[p][s] for p in qids])
            sims=Q@C.T                                         # (nq x ncat)
            top5=np.argpartition(-sims,5,axis=1)[:,:5]
            r1=r5=0
            for k,p in enumerate(qids):
                gi=pos[p]; order=top5[k][np.argsort(-sims[k,top5[k]])]
                if order[0]==gi: r1+=1
                if gi in top5[k]: r5+=1
            print(f"     {s:>4} {len(qids):>5} {r1/len(qids):>7.3f} {r5/len(qids):>7.3f}")

if __name__=="__main__":
    tags=sys.argv[1:] or ["nemotron_vl","gemini_embedding_2"]
    for t in tags: analyze(t)

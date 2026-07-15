#!/usr/bin/env python3
"""Confirm whether gemini-embedding-2-preview == gemini-embedding-2 by embedding a sample
of (photo,size) with the preview model and comparing cosine to the stored GA vectors."""
import os, json, glob, base64, numpy as np, requests
KEY=os.environ["OPENROUTER_API_KEY"]; EP="https://openrouter.ai/api/v1/embeddings"
photos={ln.split()[0]:ln.split() for ln in open("flickr_manifest.txt").read().splitlines()}
SMALL={"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX={"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}; BIGSUF={"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}
def ue(pid,size):
    p=photos[pid]
    if size in SMALL: return f"https://live.staticflickr.com/{p[1]}/{pid}_{p[2]}{SMALL[size]}.jpg","jpg"
    if size in BIGIDX:
        s=p[BIGIDX[size]]; return (f"https://live.staticflickr.com/{p[1]}/{pid}_{s}{BIGSUF[size]}.jpg","jpg") if s!="-" else (None,None)
    if size=="o":
        s,e=p[9],p[10]; return (f"https://live.staticflickr.com/{p[1]}/{pid}_{s}_o.{e}",e) if s!="-" else (None,None)
    return None,None
# load GA vectors keyed by (pid,size)
ga={}
for meta in glob.glob("emb/gemini_embedding_2/chunk_*.meta.tsv"):
    mat=np.load(meta.replace(".meta.tsv",".npy")); rows=[l.rstrip().split("\t") for l in open(meta)][1:]
    for i,r in enumerate(rows):
        if r[3]=="1": ga[(r[0],r[1])]=mat[i]
S=requests.Session(); S.headers.update({"Authorization":f"Bearer {KEY}"})
def emb(pid,size):
    url,ext=ue(pid,size); path=f"images/{pid}__{size}.{ext}"
    if os.path.exists(path):
        b=base64.b64encode(open(path,"rb").read()).decode()
        mime="png" if ext=="png" else ("gif" if ext=="gif" else "jpeg")
        body={"model":"google/gemini-embedding-2-preview","input":[{"content":[{"type":"image_url","image_url":{"url":f"data:image/{mime};base64,{b}"}}]}],"encoding_format":"float"}
    else:
        body={"model":"google/gemini-embedding-2-preview","input":[{"content":[{"type":"image_url","image_url":{"url":url}}]}],"encoding_format":"float"}
    r=S.post(EP,data=json.dumps(body),timeout=120)
    return np.array(r.json()["data"][0]["embedding"],dtype=np.float32) if r.status_code==200 else None
# sample ~60 pairs spanning sizes
keys=sorted(ga.keys())
sizes=['sq','q','t','s','n','w','m','z','c','l','h','k','3k','4k','5k','6k','o']
sample=[]
for s in sizes:
    ks=[k for k in keys if k[1]==s][:4]
    sample+=ks
coss=[]
for pid,size in sample:
    v=emb(pid,size)
    if v is None: continue
    g=ga[(pid,size)]
    c=float(np.dot(v,g)/(np.linalg.norm(v)*np.linalg.norm(g)))
    coss.append(c)
coss=np.array(coss)
print(f"compared {len(coss)} (photo,size) pairs across sizes")
print(f"cosine(preview, GA): mean={coss.mean():.6f} min={coss.min():.6f} max={coss.max():.6f}")
print("=> preview is IDENTICAL to GA" if coss.min()>0.9999 else "=> preview DIFFERS from GA (run it fully)")

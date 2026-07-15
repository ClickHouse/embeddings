#!/usr/bin/env python3
"""Embed every (photo,size) image with one OpenRouter image-embedding model.
Input via Flickr URL (server-side fetch); falls back to base64 of the local archived file.
Stores sharded, resumable: emb/<tag>/chunk_<i>.npy (float32 N x dim) + chunk_<i>.meta.tsv."""
import os, sys, json, time, base64, threading
from concurrent.futures import ThreadPoolExecutor
import requests

KEY = os.environ["OPENROUTER_API_KEY"]
EP = "https://openrouter.ai/api/v1/embeddings"
BASE = "https://live.staticflickr.com"
SMALL = {"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX = {"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}
BIGSUF = {"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}
IMGDIR = "images"
CHUNK = 1000
WORKERS = 16

def url_ext(p, size):
    pid, server, base = p[0], p[1], p[2]
    if size in SMALL: return f"{BASE}/{server}/{pid}_{base}{SMALL[size]}.jpg", "jpg"
    if size in BIGIDX:
        s = p[BIGIDX[size]]
        return (f"{BASE}/{server}/{pid}_{s}{BIGSUF[size]}.jpg","jpg") if s!="-" else (None,None)
    if size == "o":
        s,e = p[9],p[10]
        return (f"{BASE}/{server}/{pid}_{s}_o.{e}", e) if s!="-" else (None,None)
    return None,None

photos = {ln.split()[0]: ln.split() for ln in open("flickr_manifest.txt").read().splitlines()}
idxphotos = [ln.split() for ln in open("flickr_manifest.txt").read().splitlines()]
good = set()
for fn in ("results_full.tsv","results_retry.tsv"):
    f=open(fn); next(f)
    for ln in f:
        a=ln.rstrip("\n").split("\t")
        if len(a)>=4 and a[2]=="200": good.add((int(a[0]),a[1]))
# deterministic task order
tasks=[]
for idx,size in sorted(good, key=lambda x:(idxphotos[x[0]][0], x[1])):
    p=idxphotos[idx]; url,ext=url_ext(p,size)
    if url: tasks.append((p[0],size,url,ext))

tl = threading.local()
def sess():
    if not hasattr(tl,"s"):
        tl.s=requests.Session()
        tl.s.headers.update({"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    return tl.s

def _try(body, attempts, base_sleep):
    for attempt in range(attempts):
        try:
            r = sess().post(EP, data=json.dumps(body), timeout=180)
            if r.status_code==200:
                j=r.json(); return j["data"][0]["embedding"], j.get("usage",{}).get("prompt_tokens",0)
            if r.status_code in (408,429,500,502,503,504):
                ra=r.headers.get("Retry-After"); time.sleep(min(int(ra),20) if (ra and ra.isdigit()) else base_sleep*(attempt+1)); continue
            break
        except Exception:
            time.sleep(base_sleep*(attempt+1))
    return None,-1

def embed_one(model, pid, size, url, ext):
    # PRIMARY: base64 of local archived file (no Flickr fetch -> no throttling)
    path=f"{IMGDIR}/{pid}__{size}.{ext}"
    if os.path.exists(path) and os.path.getsize(path)>0:
        b=base64.b64encode(open(path,"rb").read()).decode()
        mime="png" if ext=="png" else ("gif" if ext=="gif" else "jpeg")
        body={"model":model,"input":[{"content":[{"type":"image_url","image_url":{"url":f"data:image/{mime};base64,{b}"}}]}],"encoding_format":"float"}
        v,tok=_try(body,4,2)
        if v: return v,tok
    # FALLBACK: Flickr URL (for the few un-archived images)
    body={"model":model,"input":[{"content":[{"type":"image_url","image_url":{"url":url}}]}],"encoding_format":"float"}
    return _try(body,5,3)

def main():
    import numpy as np
    model, tag = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv)>3 else len(tasks)
    use = tasks[:limit]
    outdir=f"emb/{tag}"; os.makedirs(outdir,exist_ok=True)
    nchunks=(len(use)+CHUNK-1)//CHUNK
    print(f"model={model} tag={tag} tasks={len(use)} chunks={nchunks}",flush=True)
    t0=time.time(); total_done=0; fails=0
    for ci in range(nchunks):
        npy=f"{outdir}/chunk_{ci}.npy"; meta=f"{outdir}/chunk_{ci}.meta.tsv"
        if os.path.exists(npy) and os.path.exists(meta):
            total_done+=sum(1 for _ in open(meta))-1; continue
        chunk=use[ci*CHUNK:(ci+1)*CHUNK]
        results=[None]*len(chunk)
        def work(j):
            pid,size,url,ext=chunk[j]
            v,tok=embed_one(model,pid,size,url,ext)
            return j,pid,size,v,tok
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for j,pid,size,v,tok in ex.map(lambda j: work(j), range(len(chunk))):
                results[j]=(pid,size,v,tok)
        dim=next((len(v) for _,_,v,_ in results if v),0)
        mat=np.zeros((len(chunk),dim),dtype=np.float32)
        with open(meta,"w") as mf:
            mf.write("photo_id\tsize\ttokens\tok\n")
            for j,(pid,size,v,tok) in enumerate(results):
                ok=1 if v else 0
                if not v: fails+=1
                else: mat[j]=v
                mf.write(f"{pid}\t{size}\t{tok}\t{ok}\n")
        np.save(npy,mat)
        total_done+=len(chunk)
        print(f"  chunk {ci+1}/{nchunks} done total={total_done} fails={fails} {time.time()-t0:.0f}s",flush=True)
    print(f"FINISHED {tag}: {total_done} embedded, {fails} failures, {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Re-embed any rows marked ok=0 in a model's chunks, using base64 of the local archived
image (preferred) or the Flickr URL. Patches the chunk .npy and .meta.tsv in place."""
import os, sys, json, time, base64, glob
import numpy as np, requests

KEY=os.environ["OPENROUTER_API_KEY"]; EP="https://openrouter.ai/api/v1/embeddings"
BASE="https://live.staticflickr.com"
SMALL={"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX={"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}; BIGSUF={"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}
photos={ln.split()[0]:ln.split() for ln in open("flickr_manifest.txt").read().splitlines()}
def url_ext(pid,size):
    p=photos[pid]; server=p[1]; base=p[2]
    if size in SMALL: return f"{BASE}/{server}/{pid}_{base}{SMALL[size]}.jpg","jpg"
    if size in BIGIDX:
        s=p[BIGIDX[size]]; return (f"{BASE}/{server}/{pid}_{s}{BIGSUF[size]}.jpg","jpg") if s!="-" else (None,None)
    if size=="o":
        s,e=p[9],p[10]; return (f"{BASE}/{server}/{pid}_{s}_o.{e}",e) if s!="-" else (None,None)
    return None,None

S=requests.Session(); S.headers.update({"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
def embed(model,pid,size):
    url,ext=url_ext(pid,size); path=f"images/{pid}__{size}.{ext}"
    bodies=[]
    if os.path.exists(path) and os.path.getsize(path)>0:
        b=base64.b64encode(open(path,"rb").read()).decode()
        mime="png" if ext=="png" else ("gif" if ext=="gif" else "jpeg")
        bodies.append({"model":model,"input":[{"content":[{"type":"image_url","image_url":{"url":f"data:image/{mime};base64,{b}"}}]}],"encoding_format":"float"})
    if url:
        bodies.append({"model":model,"input":[{"content":[{"type":"image_url","image_url":{"url":url}}]}],"encoding_format":"float"})
    for body in bodies:
        for attempt in range(6):
            try:
                r=S.post(EP,data=json.dumps(body),timeout=300)
                if r.status_code==200:
                    j=r.json(); return j["data"][0]["embedding"], j.get("usage",{}).get("prompt_tokens",0)
                if r.status_code in (408,429,500,502,503,504): time.sleep(3*(attempt+1)); continue
                break
            except Exception: time.sleep(3*(attempt+1))
    return None,-1

def main():
    model, tag = sys.argv[1], sys.argv[2]
    fixed=0; still=0
    for meta in sorted(glob.glob(f"emb/{tag}/chunk_*.meta.tsv")):
        npy=meta.replace(".meta.tsv",".npy")
        rows=[l.rstrip("\n").split("\t") for l in open(meta)][1:]
        bad=[i for i,r in enumerate(rows) if r[3]=="0"]
        if not bad: continue
        mat=np.load(npy)
        from concurrent.futures import ThreadPoolExecutor
        def do(i):
            pid,size=rows[i][0],rows[i][1]; return i,embed(model,pid,size)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i,(v,tok) in ex.map(do,bad):
                if v:
                    mat[i]=v; rows[i][2]=str(tok); rows[i][3]="1"; fixed+=1
                else: still+=1
        np.save(npy,mat)
        with open(meta,"w") as f:
            f.write("photo_id\tsize\ttokens\tok\n")
            for r in rows: f.write("\t".join(r)+"\n")
        print(f"  {os.path.basename(npy)}: fixed {len(bad)-sum(1 for i in bad if rows[i][3]=='0')}/{len(bad)}",flush=True)
    print(f"FIX {tag}: fixed={fixed} still_failed={still}",flush=True)

if __name__=="__main__": main()

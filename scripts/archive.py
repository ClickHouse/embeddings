#!/usr/bin/env python3
"""Re-download the 12,709 successfully-fetched images and KEEP them on disk.
Builds image_catalog.tsv: photo_id<TAB>size<TAB>url<TAB>path<TAB>bytes"""
import os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

BASE = "https://live.staticflickr.com"
SMALL = {"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX = {"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}
BIGSUF = {"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}
IMGDIR = "images"

def url_for(p, size):
    pid, server, base = p[0], p[1], p[2]
    if size in SMALL: return f"{BASE}/{server}/{pid}_{base}{SMALL[size]}.jpg", "jpg"
    if size in BIGIDX:
        s = p[BIGIDX[size]]
        return (f"{BASE}/{server}/{pid}_{s}{BIGSUF[size]}.jpg","jpg") if s!="-" else (None,None)
    if size == "o":
        s,e = p[9],p[10]
        return (f"{BASE}/{server}/{pid}_{s}_o.{e}", e) if s!="-" else (None,None)
    return None,None

photos = [ln.split() for ln in open("flickr_manifest.txt").read().splitlines()]

# the set of (photo_idx, size) that returned 200 in the survey
good = set()
for fn in ("results_full.tsv","results_retry.tsv"):
    f=open(fn); next(f)
    for ln in f:
        a=ln.rstrip("\n").split("\t")
        if len(a)>=4 and a[2]=="200": good.add((int(a[0]),a[1]))

tasks=[]
for idx,size in good:
    p=photos[idx]; url,ext=url_for(p,size)
    if url: tasks.append((p[0],size,url,ext))
print("images to archive:",len(tasks),flush=True)

os.makedirs(IMGDIR,exist_ok=True)
HEADERS={"User-Agent":"size-survey/1.0","Accept-Encoding":"identity"}
done=0; lock=threading.Lock(); cat_lock=threading.Lock()
catf=open("image_catalog.tsv","w"); catf.write("photo_id\tsize\turl\tpath\tbytes\n")

def grab(t):
    pid,size,url,ext=t
    path=f"{IMGDIR}/{pid}__{size}.{ext}"
    if os.path.exists(path) and os.path.getsize(path)>0:
        return (pid,size,url,path,os.path.getsize(path))
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers=HEADERS)
            with urllib.request.urlopen(req,timeout=120) as r:
                data=r.read()
            tmp=path+".tmp"
            with open(tmp,"wb") as fo: fo.write(data)
            os.replace(tmp,path)
            return (pid,size,url,path,len(data))
        except urllib.error.HTTPError as e:
            if e.code==429:
                import time; time.sleep(3*(attempt+1)); continue
            return None
        except Exception:
            import time; time.sleep(2*(attempt+1)); continue
    return None

with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed([ex.submit(grab,t) for t in tasks]):
        r=fut.result()
        with lock:
            done+=1
            if done%1000==0: print(f"  {done}/{len(tasks)}",flush=True)
        if r:
            with cat_lock:
                catf.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n"); catf.flush()
catf.close()
print(f"DONE archived (catalog has the successes)",flush=True)

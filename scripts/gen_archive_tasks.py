#!/usr/bin/env python3
BASE="https://live.staticflickr.com"
SMALL={"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX={"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}
BIGSUF={"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}
def ue(p,size):
    pid,server,base=p[0],p[1],p[2]
    if size in SMALL: return f"{BASE}/{server}/{pid}_{base}{SMALL[size]}.jpg","jpg"
    if size in BIGIDX:
        s=p[BIGIDX[size]]; return (f"{BASE}/{server}/{pid}_{s}{BIGSUF[size]}.jpg","jpg") if s!="-" else (None,None)
    if size=="o":
        s,e=p[9],p[10]; return (f"{BASE}/{server}/{pid}_{s}_o.{e}",e) if s!="-" else (None,None)
    return None,None
idxp=[ln.split() for ln in open("flickr_manifest.txt").read().splitlines()]
good=set()
for fn in ("results_full.tsv","results_retry.tsv"):
    f=open(fn); next(f)
    for ln in f:
        a=ln.rstrip("\n").split("\t")
        if len(a)>=4 and a[2]=="200": good.add((int(a[0]),a[1]))
n=0
with open("archive_tasks.tsv","w") as o:
    for idx,size in sorted(good,key=lambda x:(idxp[x[0]][0],x[1])):
        p=idxp[idx]; url,ext=ue(p,size)
        if url: o.write(f"{p[0]}\t{size}\t{url}\timages/{p[0]}__{size}.{ext}\n"); n+=1
print("tasks:",n)

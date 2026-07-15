#!/usr/bin/env python3
import random, statistics as st
random.seed(7)
TOTAL_PHOTOS = 217_059_448
COUNT = {'sq':217_059_447,'q':217_059_447,'t':217_059_442,'s':217_010_380,'n':216_896_222,
 'w':216_479_936,'m':216_038_479,'z':214_397_490,'c':203_397_400,'l':192_831_045,
 'h':165_291_206,'o':158_441_872,'k':147_629_778,'3k':73_293_284,'4k':51_873_272,
 '5k':24_150_939,'6k':8_427_564}
ORDER=list(COUNT); TIER=['o','6k','5k','4k','3k','k','h','l','c','z','w','m','n','s','t','q','sq']

per_photo={}
def load(p,h=True):
    f=open(p)
    if h: next(f)
    for ln in f:
        a=ln.rstrip("\n").split("\t")
        if len(a)<4 or a[2]!="200": continue
        if a[1] not in COUNT: continue
        per_photo.setdefault(a[0],{})[a[1]]=int(a[3])
load("results_full.tsv"); load("results_retry.tsv")
photos=list(per_photo)
N=len(photos)

def human(n):
    for u in ['B','KB','MB','GB','TB','PB']:
        if n<1024 or u=='PB': return f"{n:,.1f} {u}"
        n/=1024

def totals(sample):
    by={k:[] for k in ORDER}; best=[]
    for ph in sample:
        d=per_photo[ph]
        for k,v in d.items(): by[k].append(v)
        for t in TIER:
            if t in d: best.append(d[t]); break
    o_tot=COUNT['o']*st.mean(by['o'])
    all_tot=sum(COUNT[k]*st.mean(by[k]) for k in ORDER if by[k])
    best_tot=TOTAL_PHOTOS*st.mean(best)
    return o_tot, best_tot, all_tot

pt=totals(photos)
B=600
res=[[],[],[]]
for _ in range(B):
    s=[random.choice(photos) for _ in range(N)]
    for i,v in enumerate(totals(s)): res[i].append(v)
def ci(x):
    x=sorted(x); return x[int(0.025*len(x))], x[int(0.975*len(x))]
labels=["originals (o) whole-dataset","best-copy-per-photo whole-dataset","all-17-sizes mirrored"]
print("Point estimate with 95% bootstrap CI (sampling uncertainty only; nominal = assumes every record still downloadable):\n")
for lab,p,r in zip(labels,pt,res):
    lo,hi=ci(r)
    print(f"  {lab:<38}: {human(p):>10}   95% CI [{human(lo)} .. {human(hi)}]")

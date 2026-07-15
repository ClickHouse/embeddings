#!/usr/bin/env python3
import glob, os
from collections import defaultdict, Counter
from PIL import Image

# nemotron tokens keyed by (photo_id,size)
tok={}
for m in glob.glob("emb/nemotron_vl/chunk_*.meta.tsv"):
    for l in open(m).read().splitlines()[1:]:
        a=l.split("\t")
        if a[3]=="1": tok[(a[0],a[1])]=int(a[2])

def dims(pid,size,ext="jpg"):
    p=f"images/{pid}__{size}.{ext}"
    if not os.path.exists(p):
        for e in("jpg","png","gif"):
            if os.path.exists(f"images/{pid}__{size}.{e}"): p=f"images/{pid}__{size}.{e}";break
    try:
        with Image.open(p) as im: return im.size  # (w,h)
    except: return None

# 1) For size 'm' (aspect-preserving): does token count depend on aspect ratio?
print("=== nemotron tokens vs ASPECT RATIO, at fixed size 'm' (~500px long edge) ===")
buck=defaultdict(Counter)
pids=[k[0] for k in tok if k[1]=='m']
import itertools
for pid in pids[:600]:
    d=dims(pid,'m')
    if not d: continue
    w,h=d; ar=max(w,h)/min(w,h)
    b=("1.00 (square)" if ar<1.05 else "1.05-1.34" if ar<1.34 else "1.34-1.55 (3:2/4:3)" if ar<1.55 else "1.55-1.85 (16:9)" if ar<1.85 else ">1.85 (pano)")
    buck[b][tok[(pid,'m')]]+=1
for b in ["1.00 (square)","1.05-1.34","1.34-1.55 (3:2/4:3)","1.55-1.85 (16:9)",">1.85 (pano)"]:
    if b in buck:
        c=buck[b]; n=sum(c.values())
        print(f"  aspect {b:<20} n={n:<4} tokens-> {dict(c.most_common(5))}")

# 2) Same PHOTO, square crop (q=150 square) vs aspect-preserving (m) vs original
print("\n=== same photo: square crop 'q' vs aspect 'm' vs 'o' (w x h -> tokens) ===")
shown=0
for pid in pids:
    if (pid,'q') in tok and (pid,'m') in tok and (pid,'o') in tok:
        dm=dims(pid,'m'); do=dims(pid,'o','jpg')
        if not dm: continue
        print(f"  {pid}: q(square)->{tok[(pid,'q')]}  m {dm[0]}x{dm[1]}->{tok[(pid,'m')]}  o->{tok[(pid,'o')]}")
        shown+=1
        if shown>=12: break

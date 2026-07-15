#!/usr/bin/env python3
import glob, numpy as np, sys
bad=0; tot=0
for tag in ["nemotron_vl","gemini_embedding_2"]:
    for f in sorted(glob.glob(f"emb/{tag}/chunk_*.npy")):
        try:
            a=np.load(f); m=f.replace(".npy",".meta.tsv"); nrows=sum(1 for _ in open(m))-1
            tot+=1
            if a.shape[0]!=nrows: print("MISMATCH",f,a.shape,nrows); bad+=1
        except Exception as e:
            print("CORRUPT",f,e); bad+=1
print(f"checked {tot} npy files, problems: {bad}")

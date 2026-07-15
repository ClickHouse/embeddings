#!/usr/bin/env python3
"""Consolidate a model's chunked embeddings into one parquet: photo_id, size, px, tokens, embedding."""
import sys, glob, numpy as np, pyarrow as pa, pyarrow.parquet as pq
PX={'sq':75,'q':150,'t':100,'s':240,'n':320,'w':400,'m':500,'z':640,'c':800,'l':1024,
    'h':1600,'k':2048,'3k':3072,'4k':4096,'5k':5120,'6k':6144,'o':0}
def consolidate(tag):
    pids=[]; sizes=[]; pxs=[]; toks=[]; vecs=[]
    for meta in sorted(glob.glob(f"emb/{tag}/chunk_*.meta.tsv"),
                       key=lambda p:int(p.split('chunk_')[1].split('.')[0])):
        mat=np.load(meta.replace(".meta.tsv",".npy"))
        rows=[l.rstrip("\n").split("\t") for l in open(meta)][1:]
        for i,r in enumerate(rows):
            if r[3]!="1": continue
            pids.append(r[0]); sizes.append(r[1]); pxs.append(PX[r[1]])
            toks.append(int(r[2])); vecs.append(mat[i])
    arr=np.stack(vecs).astype(np.float32)
    t=pa.table({"photo_id":pids,"size":sizes,"px":pxs,"tokens":toks,
                "embedding":pa.array(list(arr), type=pa.list_(pa.float32()))})
    out=f"emb/{tag}.parquet"; pq.write_table(t,out)
    print(f"{tag}: {len(pids)} rows, dim={arr.shape[1]} -> {out}")
if __name__=="__main__":
    for tag in sys.argv[1:]: consolidate(tag)

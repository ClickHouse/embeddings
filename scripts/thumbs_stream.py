#!/usr/bin/env python3
"""stdin: parquet (md5, image=JPEG). stdout: parquet (md5, thumb Array(UInt8) 75x75x3).
Gamma-correct Lanczos downscale (sRGB->linear->resize->sRGB)."""
import sys, io, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
RES = Image.Resampling.LANCZOS; SIZE = 75
def make_thumb(jpeg):
    try: im = Image.open(io.BytesIO(jpeg)).convert('RGB')
    except Exception: return None
    a = np.asarray(im, np.float32)/255.0
    lin = np.where(a<=0.04045, a/12.92, ((a+0.055)/1.055)**2.4).astype(np.float32)
    out = np.empty((SIZE,SIZE,3), np.float32)
    for c in range(3): out[:,:,c]=np.asarray(Image.fromarray(lin[:,:,c],'F').resize((SIZE,SIZE),RES),np.float32)
    s = np.where(out<=0.0031308, out*12.92, 1.055*np.power(np.clip(out,0,1),1/2.4)-0.055)
    return np.clip(s*255.0+0.5,0,255).astype(np.uint8).reshape(-1)
data = sys.stdin.buffer.read()
t = pq.read_table(pa.BufferReader(data))
gm, gt = [], []
for m, jp in zip(t.column('md5').to_pylist(), t.column('image').to_pylist()):
    th = make_thumb(jp)
    if th is not None: gm.append(m); gt.append(th)
buf = io.BytesIO()
pq.write_table(pa.table({'md5': gm, 'thumb': pa.array(gt, type=pa.list_(pa.uint8()))}), buf)
sys.stdout.buffer.write(buf.getvalue())
sys.stderr.write(f"{len(t)}->{len(gm)}\n")

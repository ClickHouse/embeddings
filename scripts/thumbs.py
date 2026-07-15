#!/usr/bin/env python3
"""Gamma-correct 75x75 RGB thumbnails from JPEG blobs.
Reads a parquet (md5, image=JPEG bytes), resizes each image the *proper* way
(decode -> sRGB->linear light -> Lanczos downscale -> linear->sRGB -> uint8),
writes a parquet (md5, thumb = Array(UInt8), 75*75*3=16875 elements)."""
import sys, io, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor
ImageFile.LOAD_TRUNCATED_IMAGES = True
RES = Image.Resampling.LANCZOS
SIZE = 75

def make_thumb(jpeg):
    try:
        im = Image.open(io.BytesIO(jpeg)).convert('RGB')
    except Exception:
        return None
    a = np.asarray(im, dtype=np.float32) / 255.0                                  # sRGB [0,1]
    lin = np.where(a <= 0.04045, a/12.92, ((a+0.055)/1.055)**2.4).astype(np.float32)  # -> linear light
    out = np.empty((SIZE, SIZE, 3), np.float32)
    for c in range(3):                                                            # Lanczos in linear space, per channel (float)
        out[:, :, c] = np.asarray(Image.fromarray(lin[:, :, c], 'F').resize((SIZE, SIZE), RES), np.float32)
    s = np.where(out <= 0.0031308, out*12.92, 1.055*np.power(np.clip(out, 0, 1), 1/2.4) - 0.055)  # linear -> sRGB
    return np.clip(s*255.0 + 0.5, 0, 255).astype(np.uint8).reshape(-1).tobytes() # 16875 bytes, row-major HxWxC

if __name__ == '__main__':
    inp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    t = pq.read_table(inp); md5s = t.column('md5').to_pylist(); imgs = t.column('image').to_pylist()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        thumbs = list(ex.map(make_thumb, imgs, chunksize=8))
    gm, gt = [], []
    for m, th in zip(md5s, thumbs):
        if th is not None:
            gm.append(m); gt.append(np.frombuffer(th, dtype=np.uint8))
    pq.write_table(pa.table({'md5': gm, 'thumb': pa.array(gt, type=pa.list_(pa.uint8()))}), outp)
    print(f"in={len(md5s)} out={len(gm)} failed={len(md5s)-len(gm)}")

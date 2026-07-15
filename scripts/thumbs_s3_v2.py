#!/usr/bin/env python3
"""stdin: md5/line. Bounded producer/consumer: NT fetch threads (I/O only) -> queue -> main-thread resize.
image_thumbs2 variant vs image_thumbs:
  (1) CENTER-CROP to the largest centered square (keep proportions, no squashing).
  (2) SHARPER downscale: INTER_AREA prefilter to 2x target (anti-alias the big reduction) then
      INTER_LANCZOS4 finish at exactly 2x (sharp, detail-preserving) -- all in linear light.
stdout: parquet (md5, thumb Array(UInt8) 16875 = 75*75*3, row-major HWC). arg1=NT."""
import sys, io, queue, threading, numpy as np, pyarrow as pa, pyarrow.parquet as pq, cv2
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(1)                     # single-threaded cv2 per process; parallelism comes from W processes, not cv2's internal thread pool
SIZE = 75; BUCKET = "multimedia-commons"
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 64
s3 = boto3.client("s3", region_name="us-west-2",
    config=Config(signature_version=UNSIGNED, max_pool_connections=NT+8, retries={'max_attempts':5,'mode':'standard'}))
_x = np.arange(256, dtype=np.float32)/255.0
_S2L = np.where(_x <= 0.04045, _x/12.92, ((_x+0.055)/1.055)**2.4).astype(np.float32)   # sRGB -> linear LUT
def fetch(m):
    try: return m, s3.get_object(Bucket=BUCKET, Key=f"data/images/{m[:3]}/{m[3:6]}/{m}.jpg")["Body"].read()
    except Exception: return m, None
def thumb(jp):
    try: a = np.asarray(Image.open(io.BytesIO(jp)).convert('RGB'), np.uint8)
    except Exception: return None
    h, w = a.shape[:2]
    s = min(h, w); top = (h - s)//2; left = (w - s)//2
    lin = _S2L[a[top:top+s, left:left+s]]                                             # centered square crop -> linear light
    mid = cv2.resize(lin, (SIZE*2, SIZE*2), interpolation=cv2.INTER_AREA)             # area prefilter (anti-alias)
    out = cv2.resize(mid, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)             # Lanczos finish (sharpness)
    srgb = np.where(out <= 0.0031308, out*12.92, 1.055*np.power(np.clip(out, 0, 1), 1/2.4) - 0.055)
    return np.clip(srgb*255.0 + 0.5, 0, 255).astype(np.uint8).reshape(-1)
md5s = [l.strip() for l in sys.stdin if l.strip()]
in_q = queue.Queue()
for m in md5s: in_q.put(m)
for _ in range(NT): in_q.put(None)
out_q = queue.Queue(maxsize=256)
def worker():
    while True:
        m = in_q.get()
        if m is None: out_q.put(None); return
        out_q.put(fetch(m))
for _ in range(NT): threading.Thread(target=worker, daemon=True).start()
gm, gt, done = [], [], 0
while done < NT:
    it = out_q.get()
    if it is None: done += 1; continue
    m, jp = it
    if jp is None: continue
    th = thumb(jp)
    if th is not None: gm.append(m); gt.append(th)
buf = io.BytesIO(); pq.write_table(pa.table({'md5': gm, 'thumb': pa.array(gt, type=pa.list_(pa.uint8()))}), buf)
sys.stdout.buffer.write(buf.getvalue()); sys.stderr.write(f"md5s={len(md5s)} thumbs={len(gm)}\n")

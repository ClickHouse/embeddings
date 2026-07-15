#!/usr/bin/env python3
import glob, random, io, subprocess
from PIL import Image
random.seed(42)
files = sorted(glob.glob('images/*__m.jpg'))   # medium-500 == image_blob content
random.shuffle(files)
files = files[:1000]
try:
    import zstandard as zstd
    _c = zstd.ZstdCompressor(level=6)
    z6 = lambda b: len(_c.compress(b))
    backend = "zstandard lib"
except ImportError:
    def z6(b):
        return len(subprocess.run(['zstd','-6','-c'], input=b, stdout=subprocess.PIPE).stdout)
    backend = "zstd CLI"

tj = tr = tz = n = px = 0
for f in files:
    jb = open(f,'rb').read()
    try:
        im = Image.open(io.BytesIO(jb)).convert('RGB')
    except Exception:
        continue
    raw = im.tobytes()              # raw RGB, 3 bytes/pixel
    tj += len(jb); tr += len(raw); tz += z6(raw); n += 1; px += im.width*im.height

print(f"backend: {backend}   N = {n} images   avg {px/n/1e6:.3f} MP")
print(f"{'format':<16}{'total':>12}{'avg/img':>12}{'vs JPEG':>10}")
print(f"{'JPEG (stored)':<16}{tj/1e6:>10.1f} MB{tj/n/1024:>10.1f} KB{1.0:>9.2f}x")
print(f"{'raw RGB':<16}{tr/1e6:>10.1f} MB{tr/n/1024:>10.1f} KB{tr/tj:>9.2f}x")
print(f"{'RGB + ZSTD(6)':<16}{tz/1e6:>10.1f} MB{tz/n/1024:>10.1f} KB{tz/tj:>9.2f}x")
print(f"ZSTD(6) shrinks raw RGB to {tz/tr*100:.0f}% of raw; result is {tz/tj:.1f}x the JPEG size")

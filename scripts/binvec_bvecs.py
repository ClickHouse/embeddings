#!/usr/bin/env python3
"""stdin: whole .bvecs (each record = [int32 dim][dim uint8]). arg: id_offset.
stdout: ClickHouse RowBinary for (id UInt64, vec Array(UInt8))."""
import sys, numpy as np
off = int(sys.argv[1]) if len(sys.argv) > 1 else 0
data = sys.stdin.buffer.read()
dim = int(np.frombuffer(data[:4], '<u4')[0])
rec = 4 + dim
arr = np.frombuffer(data, np.uint8).reshape(-1, rec)
n = arr.shape[0]
vecs = arr[:, 4:]
def varint(x):
    b = bytearray()
    while True:
        y = x & 0x7f; x >>= 7; b.append(y | 0x80 if x else y)
        if not x: return bytes(b)
vp = np.frombuffer(varint(dim), np.uint8); vl = len(vp)
outrow = 8 + vl + dim
out = np.empty((n, outrow), np.uint8)
out[:, 0:8] = np.arange(off, off+n, dtype='<u8').view(np.uint8).reshape(n, 8)
out[:, 8:8+vl] = vp
out[:, 8+vl:] = vecs
sys.stdout.buffer.write(out.tobytes())
sys.stderr.write(f"n={n} dim={dim}\n")

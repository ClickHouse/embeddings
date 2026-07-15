#!/usr/bin/env python3
"""stdin: whole range-search GT [i32 nq][i32 total][i32 nres[nq]][i32 ids[total]][f32 dists[total]].
stdout: RowBinary (query_id UInt32, neighbors Array(UInt32), distances Array(Float32)) — variable length per query."""
import sys, numpy as np
data = sys.stdin.buffer.read()
nq, total = np.frombuffer(data[:8], '<i4')
nq = int(nq); total = int(total); off = 8
nres = np.frombuffer(data, '<i4', nq, off); off += nq*4
ids = np.frombuffer(data, '<i4', total, off).astype('<u4'); off += total*4
dists = np.frombuffer(data, '<f4', total, off)
def varint(x):
    b = bytearray()
    while True:
        y = x & 0x7f; x >>= 7; b.append(y | 0x80 if x else y)
        if not x: return bytes(b)
out = bytearray(); pos = 0; w = sys.stdout.buffer
for i in range(nq):
    L = int(nres[i]); v = varint(L)
    out += i.to_bytes(4, 'little')
    out += v; out += ids[pos:pos+L].tobytes()
    out += v; out += dists[pos:pos+L].tobytes()
    pos += L
    if len(out) > (8 << 20): w.write(out); out = bytearray()
w.write(out)
sys.stderr.write(f"nq={nq} total={total}\n")

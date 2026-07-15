#!/usr/bin/env python3
"""stdin: uncompressed CSR: [i64 nrow][i64 ncol][i64 nnz][i64 indptr[nrow+1]][i32 indices[nnz]][f32 data[nnz]].
stdout: RowBinary for (id UInt64, indices Array(UInt32), values Array(Float32)), one row per sparse vector."""
import sys, numpy as np
f = sys.stdin.buffer
def rd(n):
    b = bytearray()
    while len(b) < n:
        c = f.read(n - len(b))
        if not c: break
        b += c
    return bytes(b)
nrow, ncol, nnz = np.frombuffer(rd(24), dtype='<i8')
indptr = np.frombuffer(rd((int(nrow)+1)*8), dtype='<i8')
indices = np.frombuffer(rd(int(nnz)*4), dtype='<i4')
data = np.frombuffer(rd(int(nnz)*4), dtype='<f4')
sys.stderr.write(f"nrow={nrow} ncol={ncol} nnz={nnz}\n")
def varint(n):
    b = bytearray()
    while True:
        x = n & 0x7f; n >>= 7; b.append(x | 0x80 if n else x)
        if not n: return bytes(b)
ind_u = indices.astype('<u4'); dat = data.astype('<f4')
out = sys.stdout.buffer
buf = bytearray()
for i in range(int(nrow)):
    s = int(indptr[i]); e = int(indptr[i+1]); L = e - s
    buf += i.to_bytes(8, 'little')
    v = varint(L)
    buf += v; buf += ind_u[s:e].tobytes()
    buf += v; buf += dat[s:e].tobytes()
    if len(buf) > (8 << 20):
        out.write(buf); buf = bytearray()
out.write(buf)

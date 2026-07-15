#!/usr/bin/env python3
"""stdin: raw fixed-size vectors (NO 8-byte header; caller strips it via byte range).
args: <u8|i8|f32> <dim> <id_offset>
stdout: ClickHouse RowBinary for (id UInt64, vec Array(T)) where T = UInt8/Int8/Float32.
Vectorized per block. ids are global row indices starting at id_offset."""
import sys, numpy as np
typ, dim, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
esz = {'u8': 1, 'i8': 1, 'f32': 4}[typ]
rowbytes = dim * esz

def varint(n):
    b = bytearray()
    while True:
        x = n & 0x7f; n >>= 7
        b.append(x | 0x80 if n else x)
        if not n: return bytes(b)
vpref = varint(dim)                 # RowBinary Array length prefix (constant, fixed dim)
prewidth = 8 + len(vpref)
outrow = prewidth + rowbytes
vpref_arr = np.frombuffer(vpref, dtype=np.uint8)

BR = 65536                          # rows per block
stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
leftover = b''
idx = off
while True:
    chunk = stdin.read(rowbytes * BR)
    data = leftover + chunk
    m = len(data) // rowbytes
    leftover = data[m * rowbytes:]
    if m:
        vecs = np.frombuffer(data[:m * rowbytes], dtype=np.uint8).reshape(m, rowbytes)
        out = np.empty((m, outrow), dtype=np.uint8)
        out[:, 0:8] = np.arange(idx, idx + m, dtype='<u8').view(np.uint8).reshape(m, 8)
        out[:, 8:prewidth] = vpref_arr
        out[:, prewidth:] = vecs
        stdout.write(out.tobytes())
        idx += m
    if not chunk:
        break
sys.stderr.write(f"rows={idx-off}\n")

#!/usr/bin/env python3
"""stdin: whole groundtruth file [uint32 nq][uint32 K][nq*K uint32 ids][nq*K float32 dists].
stdout: RowBinary for (query_id UInt32, neighbors Array(UInt32), distances Array(Float32)), one row/query."""
import sys, struct, numpy as np
data = sys.stdin.buffer.read()
nq, K = struct.unpack('<II', data[:8])
off = 8
ids = np.frombuffer(data, np.uint32, nq * K, off).reshape(nq, K); off += nq * K * 4
dists = np.frombuffer(data, np.float32, nq * K, off).reshape(nq, K)
def varint(n):
    b = bytearray()
    while True:
        x = n & 0x7f; n >>= 7; b.append(x | 0x80 if n else x)
        if not n: return bytes(b)
vK = np.frombuffer(varint(K), np.uint8); vl = len(vK)
rw = 4 + vl + K * 4 + vl + K * 4
out = np.empty((nq, rw), np.uint8)
out[:, 0:4] = np.arange(nq, dtype='<u4').view(np.uint8).reshape(nq, 4)
p = 4
out[:, p:p+vl] = vK; p += vl
out[:, p:p+K*4] = ids.view(np.uint8).reshape(nq, K * 4); p += K * 4
out[:, p:p+vl] = vK; p += vl
out[:, p:p+K*4] = dists.view(np.uint8).reshape(nq, K * 4)
sys.stdout.buffer.write(out.tobytes())
sys.stderr.write(f"nq={nq} K={K}\n")

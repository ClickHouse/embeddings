#!/usr/bin/env python3
"""stdin: a whole .npy shard (fp16 <f2>, shape (n, dim)) incl header.
args: <id_offset>
stdout: ClickHouse RowBinary for (id UInt64, vec Array(BFloat16)). fp16 -> bf16 (round-to-nearest-even)."""
import sys, ast, numpy as np
off = int(sys.argv[1])
data = sys.stdin.buffer.read()
assert data[:6] == b'\x93NUMPY', "not a npy file"
major = data[6]
if major == 1:
    hlen = int.from_bytes(data[8:10], 'little'); hstart = 10
else:
    hlen = int.from_bytes(data[8:12], 'little'); hstart = 12
hdr = ast.literal_eval(data[hstart:hstart+hlen].decode('latin1').strip())
n, dim = hdr['shape']
assert hdr['descr'] in ('<f2', '|f2', 'float16'), hdr['descr']
dstart = hstart + hlen
arr = np.frombuffer(data, dtype=np.float16, count=n*dim, offset=dstart).reshape(n, dim)
bits = arr.astype(np.float32).view(np.uint32)                    # fp16 -> fp32 bits
bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2')  # -> bf16, round-to-nearest-even
def varint(x):
    b = bytearray()
    while True:
        y = x & 0x7f; x >>= 7; b.append(y | 0x80 if x else y)
        if not x: return bytes(b)
vp = np.frombuffer(varint(dim), np.uint8); vl = len(vp)
outrow = 8 + vl + dim*2
out = np.empty((n, outrow), np.uint8)
out[:, 0:8] = np.arange(off, off+n, dtype='<u8').view(np.uint8).reshape(n, 8)
out[:, 8:8+vl] = vp
out[:, 8+vl:] = bf16.view(np.uint8).reshape(n, dim*2)
sys.stdout.buffer.write(out.tobytes())
sys.stderr.write(f"n={n} dim={dim}\n")

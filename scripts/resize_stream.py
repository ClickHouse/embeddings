#!/usr/bin/env python3
# Stream-resize one url-range of web_screenshots into web_thumbs (64x64 gamma-correct RGB).
# Reads RowBinary (url String, screenshot Array(UInt8)) from stdin (a `curl --compressed ... FORMAT RowBinary`
# pipe), parses incrementally (screenshots are 4.9 MB, never buffers the whole range), resizes, batch-INSERTs.
# On clean EOF it touches the marker (argv[1]) so the orchestrator marks the range done. argv[2]=label for logs.
import sys, os, time, ssl, urllib.parse, urllib.request, numpy as np, cv2
cv2.setNumThreads(1)
HOST = "https://hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:8443"
DP   = os.environ.get("CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD", "")
CTX  = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"   # bash writes the done-marker on full-pipeline success
FLUSH = 2000
_s = np.arange(256) / 255.0
LIN = np.where(_s <= 0.04045, _s / 12.92, ((_s + 0.055) / 1.055) ** 2.4).astype(np.float32)  # sRGB->linear
def delin(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(l, 1/2.4) - 0.055)             # linear->sRGB
def wv(n):
    o = bytearray()
    while True:
        b = n & 0x7f; n >>= 7; o.append(b | (0x80 if n else 0))
        if not n: return bytes(o)
def insert(payload):
    if not payload: return
    q = urllib.parse.quote("INSERT INTO default.web_thumbs (url, thumb) FORMAT RowBinary")
    url = f"{HOST}/?user=default&password={urllib.parse.quote(DP)}&query={q}"
    for a in range(6):
        try:
            urllib.request.urlopen(urllib.request.Request(url, data=payload, method="POST"), context=CTX, timeout=900).read(); return
        except Exception:
            if a == 5: raise
            time.sleep(3 * (a + 1))

inp = sys.stdin.buffer
buf = bytearray(); pos = 0
def fill(n):
    global buf, pos
    while len(buf) - pos < n:
        c = inp.read(1 << 22)
        if not c: return False
        if pos: del buf[:pos]; pos = 0
        buf += c
    return True
def rvarint():
    global pos
    r = s = 0
    while True:
        if not fill(1): return None
        x = buf[pos]; pos += 1; r |= (x & 0x7f) << s
        if not x & 0x80: return r
        s += 7
def take(n):
    global pos
    if not fill(n): return None
    b = bytes(buf[pos:pos+n]); pos += n; return b

payload = bytearray(); cnt = 0; done = 0
while True:
    ul = rvarint()
    if ul is None: break
    url = take(ul)
    sl = rvarint()
    if sl is None: break
    shot = take(sl)
    if shot is None: break
    done += 1
    if sl != 4915200: continue                       # skip partial/failed captures
    arr = np.frombuffer(shot, np.uint8).reshape(1280, 1280, 3)
    small = cv2.resize(LIN[arr], (64, 64), interpolation=cv2.INTER_AREA)   # gamma-correct box downscale
    th = np.clip(delin(small) * 255 + 0.5, 0, 255).astype(np.uint8).tobytes()
    payload += wv(ul) + url + wv(len(th)) + th; cnt += 1
    if cnt >= FLUSH:
        insert(bytes(payload)); payload = bytearray(); cnt = 0
insert(bytes(payload))
sys.stderr.write(f"[{LABEL}] done {done} rows\n")

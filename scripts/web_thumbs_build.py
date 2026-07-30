#!/usr/bin/env python3
# Build default.web_thumbs (url, thumb Array(UInt8)) = 75x75 gamma-correct RGB thumbnails of the copied
# full-res screenshots in default.web_screenshots (1280x1280 raw RGB). Client-side resize on this box
# (screenshots are already raw RGB in the DB -> no decode; SQL resize is too slow, so we pull + resize here).
#
# Parallel by url PK-range (each screenshot read exactly once): P disjoint ranges (OFFSET-computed boundaries),
# each worker cursor-paginates its range (url > cursor ORDER BY url LIMIT BATCH), gamma-correct cv2 INTER_AREA
# downscale to 75x75, batch-INSERT (url, thumb) via RowBinary. Resumable via per-range cursor files.
#
#   CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD must be set.  Run:  python3 web_thumbs_build.py [P] [BATCH]
import os, sys, time, ssl, urllib.parse, urllib.request, numpy as np, cv2
from multiprocessing import Process
cv2.setNumThreads(1)   # one resize thread per worker process (avoid fork thread-oversubscription)

HOST = "https://hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:8443"
DP   = os.environ.get("CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD", "")
P     = int(sys.argv[1]) if len(sys.argv) > 1 else 32
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 30    # screenshots fetched per query (~0.047 GiB each + 0.4)
FLUSH = 2000                                             # rows accumulated per INSERT (limit part count)
STATE = "/tmp/webthumbs"; os.makedirs(STATE, exist_ok=True)
LOG   = "/tmp/web_thumbs.log"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

# sRGB <-> linear LUTs (gamma-correct downscale so thin white/text pages don't darken)
_s = np.arange(256) / 255.0
LIN = np.where(_s <= 0.04045, _s / 12.92, ((_s + 0.055) / 1.055) ** 2.4).astype(np.float32)
def delin(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(l, 1/2.4) - 0.055)

def log(m):
    with open(LOG, "a") as f: f.write(f"{time.strftime('%H:%M:%S')} {m}\n")

def http(sql=None, body=None, fmt=None, params="", tries=5):
    extra = f"user=default&password={urllib.parse.quote(DP)}&max_memory_usage=2500000000{params}"
    if fmt: extra += f"&default_format={fmt}"
    if body is not None:  # INSERT: query in URL, RowBinary payload in body
        extra += f"&query={urllib.parse.quote(sql)}"; data = body
    else:
        data = sql.encode()
    for a in range(tries):
        try:
            req = urllib.request.Request(f"{HOST}/?{extra}", data=data, method="POST")
            with urllib.request.urlopen(req, context=CTX, timeout=1800) as r:
                return r.read()
        except Exception as e:
            msg = (e.read()[:200].decode("utf8", "replace") if hasattr(e, "read") else str(e)[:200])
            if a == tries - 1:
                log(f"HTTP give-up ({tries}x): {msg}"); raise
            time.sleep(3 * (a + 1))

def rv(b, o):           # read LEB128 varint
    r = s = 0
    while True:
        x = b[o]; o += 1; r |= (x & 0x7f) << s
        if not x & 0x80: break
        s += 7
    return r, o
def wv(n):              # write LEB128 varint
    out = bytearray()
    while True:
        b = n & 0x7f; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: break
    return bytes(out)

def resize_one(shot):
    if len(shot) != 4915200: return None
    arr = np.frombuffer(shot, np.uint8).reshape(1280, 1280, 3)
    small = cv2.resize(LIN[arr], (75, 75), interpolation=cv2.INTER_AREA)  # downscale in linear light
    return np.clip(delin(small) * 255 + 0.5, 0, 255).astype(np.uint8).tobytes()

def worker(i, hi):
    curf = f"{STATE}/cur_{i}"
    cursor = open(curf).read() if os.path.exists(curf) else ""
    hicond = "" if hi is None else f" AND url <= {esc(hi)}"
    buf = bytearray(); bufn = [0]; pend = [cursor]; done = 0
    def flush():                          # persist cursor ONLY for rows actually inserted (safe resume)
        nonlocal buf, cursor
        if bufn[0]:
            http(sql="INSERT INTO default.web_thumbs (url, thumb) FORMAT RowBinary", body=bytes(buf))
            buf = bytearray(); bufn[0] = 0
        cursor = pend[0]
        with open(curf, "w") as f: f.write(cursor)
    while True:
        sql = (f"SELECT url, screenshot FROM default.web_screenshots "
               f"WHERE url > {esc(pend[0])}{hicond} ORDER BY url LIMIT {BATCH} FORMAT RowBinary")
        raw = http(sql=sql)
        if not raw: break
        o = 0; n = 0
        while o < len(raw):
            ul, o = rv(raw, o); url = raw[o:o+ul]; o += ul
            sl, o = rv(raw, o); shot = raw[o:o+sl]; o += sl
            pend[0] = url.decode("utf8", "replace"); n += 1
            th = resize_one(shot)
            if th is None: continue
            buf += wv(len(url)) + url + wv(len(th)) + th; bufn[0] += 1
        done += n
        if bufn[0] >= FLUSH: flush()
        if n < BATCH: break
    flush()
    log(f"worker {i} DONE {done} rows")

def esc(s):  # single-quote SQL string literal
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"

def boundaries():
    total = int(http(sql="SELECT count() FROM default.web_screenshots", fmt="TSV").decode().strip())
    step = max(1, total // P)
    # one ordered scan of the (small) url column, bucketed by row/step -> first url of each bucket
    sql = (f"SELECT any(url) FROM (SELECT url, intDiv(rowNumberInAllBlocks(), {step}) AS bk "
           f"FROM (SELECT url FROM default.web_screenshots ORDER BY url)) GROUP BY bk ORDER BY bk FORMAT TSV")
    urls = http(sql=sql).decode("utf8", "replace").splitlines()
    return total, urls[1:P]   # P-1 upper bounds; worker 0 starts at "", last worker unbounded

if __name__ == "__main__":
    if not DP: sys.exit("CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD empty")
    total, bs = boundaries()
    log(f"=== web_thumbs start total={total} P={P} BATCH={BATCH} ===")
    # ranges: worker i owns (bs[i-1], bs[i]] ; first has no lower, last has no upper
    procs = []
    lowers = [""] + bs               # cursor lower bound (exclusive) per worker; worker0 starts at ""
    uppers = bs + [None]             # inclusive upper per worker; last unbounded
    for i in range(P):
        # seed cursor file with the lower bound if not already past it
        curf = f"{STATE}/cur_{i}"
        if not os.path.exists(curf):
            open(curf, "w").write(lowers[i])
        p = Process(target=worker, args=(i, uppers[i])); p.start(); procs.append(p)
    for p in procs: p.join()
    log("=== web_thumbs ALL DONE ===")

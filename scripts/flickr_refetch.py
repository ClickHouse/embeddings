#!/usr/bin/env python3
"""Re-fetch the (photo,size) requests that were rate-limited (429) or had network errors (-1),
using low concurrency + backoff so the sample (esp. originals) isn't biased."""
import sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

BASE = "https://live.staticflickr.com"
SMALL = {"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX = {"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}
BIGSUF = {"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}

def url_for(p, size):
    pid, server, base = p[0], p[1], p[2]
    if size in SMALL:
        return f"{BASE}/{server}/{pid}_{base}{SMALL[size]}.jpg"
    if size in BIGIDX:
        sec = p[BIGIDX[size]]
        return None if sec == "-" else f"{BASE}/{server}/{pid}_{sec}{BIGSUF[size]}.jpg"
    if size == "o":
        sec, ext = p[9], p[10]
        return None if sec == "-" else f"{BASE}/{server}/{pid}_{sec}_o.{ext}"
    return None

HEADERS = {"User-Agent": "size-survey/1.0", "Accept-Encoding": "identity"}
done = 0; lock = threading.Lock(); total = 0

def fetch(task):
    i, size, url = task
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=180) as r:
                n = 0
                while True:
                    c = r.read(262144)
                    if not c: break
                    n += len(c)
                return (i, size, 200, n, round(time.time()-t0, 3))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                wait = int(ra) if (ra and ra.isdigit()) else (5*(attempt+1))
                time.sleep(min(wait, 30) + 0.3*attempt)
                continue
            return (i, size, e.code, 0, 0)
        except Exception:
            time.sleep(2*(attempt+1)); continue
    return (i, size, 429, 0, 0)   # still throttled after all attempts

def main():
    manifest, results_in, out, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    global total, done
    photos = {}
    with open(manifest) as f:
        for i, ln in enumerate(f):
            photos[i] = ln.split()
    tasks = []
    with open(results_in) as f:
        next(f)
        for ln in f:
            ph, size, code, *_ = ln.rstrip("\n").split("\t")
            if code in ("429", "-1"):
                u = url_for(photos[int(ph)], size)
                if u: tasks.append((int(ph), size, u))
    total = len(tasks)
    print(f"re-fetching {total} throttled/failed requests with {workers} workers", flush=True)
    t0 = time.time()
    with open(out, "w") as fo, ThreadPoolExecutor(max_workers=workers) as ex:
        fo.write("photo\tsize\tcode\tbytes\tsecs\n")
        for fut in as_completed([ex.submit(fetch, t) for t in tasks]):
            i, size, code, n, dt = fut.result()
            fo.write(f"{i}\t{size}\t{code}\t{n}\t{dt}\n"); fo.flush()
            with lock:
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total} ({time.time()-t0:.0f}s)", flush=True)
    print(f"FINISHED {done}/{total} in {time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == "__main__":
    main()

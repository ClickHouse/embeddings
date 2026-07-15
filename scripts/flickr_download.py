#!/usr/bin/env python3
"""Download every available Flickr size for each sampled photo and record real byte sizes.
Reconstructs URLs from the compact manifest (id server base h k 3k 4k 5k 6k o oext)."""
import sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

BASE = "https://live.staticflickr.com"
# size-key -> URL suffix for the 10 sizes sharing the photo's base secret
SMALL = [("sq","_s"),("q","_q"),("t","_t"),("s","_m"),("n","_n"),
         ("w","_w"),("m",""),("z","_z"),("c","_c"),("l","_b")]
# big sizes: each has its own secret (field index in manifest), suffix, fixed ext
BIG = [("h",3,"_h","jpg"),("k",4,"_k","jpg"),("3k",5,"_3k","jpg"),
       ("4k",6,"_4k","jpg"),("5k",7,"_5k","jpg"),("6k",8,"_6k","jpg")]
# "o" handled specially (field 9 = secret, field 10 = ext)

def build_tasks(path):
    tasks = []  # (photo_idx, size_key, url)
    with open(path) as f:
        for i, ln in enumerate(f):
            p = ln.split()
            if len(p) != 11:
                continue
            pid, server, base = p[0], p[1], p[2]
            prefix = f"{BASE}/{server}/{pid}_{base}"
            for key, suf in SMALL:
                tasks.append((i, key, f"{prefix}{suf}.jpg"))
            for key, idx, suf, ext in BIG:
                sec = p[idx]
                if sec != "-":
                    tasks.append((i, key, f"{BASE}/{server}/{pid}_{sec}{suf}.{ext}"))
            o_sec, o_ext = p[9], p[10]
            if o_sec != "-" and o_ext != "-":
                tasks.append((i, key:="o", f"{BASE}/{server}/{pid}_{o_sec}_o.{o_ext}"))
    return tasks

HEADERS = {"User-Agent": "size-survey/1.0", "Accept-Encoding": "identity"}
done = 0
lock = threading.Lock()
total = 0

def fetch(task):
    i, key, url = task
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                n = 0
                while True:
                    chunk = r.read(262144)
                    if not chunk:
                        break
                    n += len(chunk)
                dt = time.time() - t0
                return (i, key, 200, n, round(dt, 3))
        except urllib.error.HTTPError as e:
            return (i, key, e.code, 0, 0)          # 404 etc. -> no retry
        except Exception:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return (i, key, -1, 0, 0)              # network failure

def main():
    manifest, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    global total, done
    tasks = build_tasks(manifest)
    total = len(tasks)
    print(f"photos in manifest, total size-requests to make: {total}", flush=True)
    t_start = time.time()
    with open(out, "w") as fo, ThreadPoolExecutor(max_workers=workers) as ex:
        fo.write("photo\tsize\tcode\tbytes\tsecs\n")
        futs = [ex.submit(fetch, t) for t in tasks]
        for fut in as_completed(futs):
            i, key, code, n, dt = fut.result()
            fo.write(f"{i}\t{key}\t{code}\t{n}\t{dt}\n")
            with lock:
                done += 1
                if done % 2000 == 0:
                    el = time.time() - t_start
                    print(f"  {done}/{total} done  ({el:.0f}s, {done/el:.0f} req/s)", flush=True)
    print(f"FINISHED {done}/{total} in {time.time()-t_start:.0f}s -> {out}", flush=True)

if __name__ == "__main__":
    main()

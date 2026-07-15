#!/usr/bin/env python3
BASE = "https://live.staticflickr.com"
SMALL = {"sq":"_s","q":"_q","t":"_t","s":"_m","n":"_n","w":"_w","m":"","z":"_z","c":"_c","l":"_b"}
BIGIDX = {"h":3,"k":4,"3k":5,"4k":6,"5k":7,"6k":8}
BIGSUF = {"h":"_h","k":"_k","3k":"_3k","4k":"_4k","5k":"_5k","6k":"_6k"}

def url_for(p, size):
    pid, server, base = p[0], p[1], p[2]
    if size in SMALL:
        return f"{BASE}/{server}/{pid}_{base}{SMALL[size]}.jpg"
    if size in BIGIDX:
        s = p[BIGIDX[size]]
        return None if s == "-" else f"{BASE}/{server}/{pid}_{s}{BIGSUF[size]}.jpg"
    if size == "o":
        s, e = p[9], p[10]
        return None if s == "-" else f"{BASE}/{server}/{pid}_{s}_o.{e}"
    return None

photos = [ln.split() for ln in open("flickr_manifest.txt").read().splitlines()]
n = 0
with open("refetch_tasks.tsv", "w") as out:
    f = open("results_full.tsv"); next(f)
    for ln in f:
        parts = ln.rstrip("\n").split("\t")
        ph, size, code = parts[0], parts[1], parts[2]
        if code in ("429", "-1"):
            u = url_for(photos[int(ph)], size)
            if u:
                out.write(f"{ph}\t{size}\t{u}\n"); n += 1
print("refetch tasks:", n)

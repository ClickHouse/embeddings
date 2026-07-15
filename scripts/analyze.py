#!/usr/bin/env python3
import statistics as st

# --- exact global availability counts (media='photo'), from ClickHouse ---
TOTAL_PHOTOS = 217_059_448
COUNT = {
 'sq':217_059_447,'q':217_059_447,'t':217_059_442,'s':217_010_380,'n':216_896_222,
 'w':216_479_936,'m':216_038_479,'z':214_397_490,'c':203_397_400,'l':192_831_045,
 'h':165_291_206,'o':158_441_872,'k':147_629_778,'3k':73_293_284,'4k':51_873_272,
 '5k':24_150_939,'6k':8_427_564}
# canonical Flickr longest-edge (px); 'o' = original (variable)
DIM = {'sq':'75²','q':'150²','t':'100','s':'240','n':'320','w':'400','m':'500','z':'640',
       'c':'800','l':'1024','h':'1600','k':'2048','3k':'3072','4k':'4096','5k':'5120',
       '6k':'6144','o':'orig'}
ORDER = ['sq','q','t','s','n','w','m','z','c','l','h','k','3k','4k','5k','6k','o']
TIER_BEST = ['o','6k','5k','4k','3k','k','h','l','c','z','w','m','n','s','t','q','sq']  # best->worst

# --- load merged 200-only measurements ---
per_size = {k: [] for k in ORDER}
per_photo = {}      # photo -> {size: bytes}
alive = set()
def load(path, has_header=True):
    f = open(path);
    if has_header: next(f)
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4: continue
        photo, size, code, byts = p[0], p[1], p[2], p[3]
        if code != "200": continue
        b = int(byts)
        if size not in per_size: continue
        per_size[size].append(b)
        per_photo.setdefault(photo, {})[size] = b
        if size == 'sq': alive.add(photo)
load("results_full.tsv")
load("results_retry.tsv")

SAMPLE_PHOTOS = 1000
ALIVE_RATE = len(alive)/SAMPLE_PHOTOS

def human(n):
    for u in ['B','KB','MB','GB','TB','PB']:
        if n < 1024 or u=='PB': return f"{n:,.1f} {u}"
        n/=1024

print(f"# Flickr dataset image-size survey")
print(f"sample photos = {SAMPLE_PHOTOS}, total successful size-downloads = {sum(len(v) for v in per_size.values()):,}")
print(f"alive photos in sample (sq returned 200) = {len(alive)}/{SAMPLE_PHOTOS} = {ALIVE_RATE*100:.1f}%  (rest deleted from Flickr)\n")

print(f"{'size':>4} {'px':>6} {'n':>5} {'mean':>10} {'median':>10} {'p90':>10} {'max':>10}  {'have%':>6}  {'NOMINAL total':>14}  {'realistic':>12}")
print("-"*120)
nominal_all = 0.0
rows = []
for k in ORDER:
    v = per_size[k]
    if not v: continue
    mean = st.mean(v); med = st.median(v)
    p90 = sorted(v)[int(0.9*len(v))-1]; mx = max(v)
    cnt = COUNT[k]
    nominal = cnt * mean                 # all recorded photos that have size k
    realistic = nominal * ALIVE_RATE     # adjust for ~deleted photos
    nominal_all += nominal
    havpct = 100*cnt/TOTAL_PHOTOS
    rows.append((k, mean, med, cnt, nominal, realistic))
    print(f"{k:>4} {DIM[k]:>6} {len(v):>5} {human(mean):>10} {human(med):>10} {human(p90):>10} {human(mx):>10}  {havpct:>5.1f}%  {human(nominal):>14}  {human(realistic):>12}")

print("-"*120)
print(f"{'ALL 17 sizes mirrored (sum of column NOMINAL)':>70}: {human(nominal_all):>14}   realistic {human(nominal_all*ALIVE_RATE)}")

# --- best copy per photo (prefer original, else largest derivative present) ---
best_bytes = []
for ph, sizes in per_photo.items():
    for t in TIER_BEST:
        if t in sizes:
            best_bytes.append(sizes[t]); break
mean_best = st.mean(best_bytes)
# every photo has at least sq, so "best copy" exists for all alive photos
best_total_nominal = TOTAL_PHOTOS * mean_best          # assumes all downloadable
best_total_real    = TOTAL_PHOTOS * ALIVE_RATE * mean_best
print()
print(f"BEST-COPY-PER-PHOTO (original if present, else largest available):")
print(f"  mean per photo = {human(mean_best)}  (median {human(st.median(best_bytes))})")
print(f"  whole dataset NOMINAL = {human(best_total_nominal)}   realistic = {human(best_total_real)}")

# --- handy single-size scenarios ---
print("\nSCENARIOS — download the WHOLE dataset at one size (nominal / realistic):")
for k,label in [('sq','square 75 thumb'),('q','square 150'),('m','medium 500'),
                ('l','large 1024'),('h','1600'),('k','2048'),('o','original')]:
    if per_size[k]:
        mean = st.mean(per_size[k]); nominal = COUNT[k]*mean
        print(f"  {label:<18} ({DIM[k]:>5}px, {100*COUNT[k]/TOTAL_PHOTOS:4.0f}% have it): "
              f"{human(nominal):>12}  /  {human(nominal*ALIVE_RATE):>12}")

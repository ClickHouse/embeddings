#!/usr/bin/env python3
# Train an NSFW logistic-regression classifier on the SigLIP2 image embeddings of WonderfulWeb.
# Positives = domain-labeled adult sites; negatives = random sample (NSFW is a small fraction -> ~all SFW).
import os, ssl, urllib.request, urllib.parse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
HOST = "https://hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:8443"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
ADULT = ("url LIKE '%pornhub%' OR url LIKE '%xvideos%' OR url LIKE '%xnxx%' OR url LIKE '%xhamster%' OR "
         "url LIKE '%redtube%' OR url LIKE '%youporn%' OR url LIKE '%spankbang%' OR url LIKE '%brazzers%' OR "
         "url LIKE '%onlyfans%' OR url LIKE '%chaturbate%' OR url LIKE '%hentai%' OR url LIKE '%.xxx%' OR url LIKE '%porn%'")

def fetch(where, limit):
    sql = f"SELECT CAST(embedding,'Array(Float32)') FROM default.web_emb_img_siglip2 WHERE {where} LIMIT {limit} FORMAT RowBinary"
    url = f"{HOST}/?user=website&password=&use_query_cache=0"
    data = urllib.request.urlopen(urllib.request.Request(url, data=sql.encode(), method="POST"), context=CTX, timeout=300).read()
    o = 0; rows = []
    while o < len(data):
        n = 0; s = 0                       # LEB128 varint (array length)
        while True:
            b = data[o]; o += 1; n |= (b & 0x7f) << s
            if not b & 0x80: break
            s += 7
        rows.append(np.frombuffer(data[o:o + n*4], dtype='<f4')); o += n*4
    return np.array(rows, dtype=np.float32)

print("fetching positives (domain-labeled adult)…")
pos = fetch(f"({ADULT}) AND cityHash64(url) % 3 = 0", 4000)
print("fetching negatives (random sample, non-adult)…")
neg = fetch(f"NOT ({ADULT}) AND cityHash64(url) % 1189 = 0", 15000)
print(f"pos={len(pos)} neg={len(neg)} dim={pos.shape[1]}")

X = np.vstack([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
print(f"held-out AUC={roc_auc_score(yte, p):.4f}  acc@0.5={accuracy_score(yte, p>0.5):.4f}")
W = clf.coef_[0].astype(np.float64); b = float(clf.intercept_[0])
# emit W (comma list) + b for the ClickHouse dotProduct scoring
with open('/tmp/nsfw_w.txt', 'w') as f:
    f.write(",".join(f"{w:.6g}" for w in W))
with open('/tmp/nsfw_b.txt', 'w') as f:
    f.write(f"{b:.6g}")
print(f"b={b:.6g}  |W|={np.linalg.norm(W):.3f}  wrote /tmp/nsfw_w.txt")

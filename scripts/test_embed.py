#!/usr/bin/env python3
import os, json, base64, urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
EP = "https://openrouter.ai/api/v1/embeddings"
MODELS = ["nvidia/llama-nemotron-embed-vl-1b-v2:free",
          "google/gemini-embedding-2",
          "google/gemini-embedding-2-preview"]

# a small, known image: photo 10080113335 sq
img_url = "https://live.staticflickr.com/5468/10080113335_e863717c76_s.jpg"

def post(body):
    req = urllib.request.Request(EP, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:
        return -1, {"err": str(e)}

def summarize(tag, status, j):
    if status == 200 and "data" in j:
        emb = j["data"][0]["embedding"]
        usage = j.get("usage", {})
        print(f"  {tag}: OK dim={len(emb)} usage={usage} head={emb[:3]}")
    else:
        print(f"  {tag}: status={status} resp={json.dumps(j)[:300]}")

# fetch image bytes once for base64 test
raw = urllib.request.urlopen(img_url, timeout=60).read()
b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
print(f"image bytes={len(raw)}")

for m in MODELS:
    print(f"\n=== {m} ===")
    body_url = {"model": m, "input":[{"content":[{"type":"image_url","image_url":{"url":img_url}}]}], "encoding_format":"float"}
    summarize("URL ", *post(body_url))
    body_b64 = {"model": m, "input":[{"content":[{"type":"image_url","image_url":{"url":b64}}]}], "encoding_format":"float"}
    summarize("B64 ", *post(body_b64))

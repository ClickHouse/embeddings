#!/usr/bin/env python3
"""Do image-embedding models read text in screenshots? Render 8 text 'documents' as images,
then check whether a paraphrased TEXT query retrieves the matching DOCUMENT IMAGE (cross-modal)."""
import os, glob, json, base64, textwrap, numpy as np, requests
from PIL import Image, ImageDraw, ImageFont
KEY=os.environ["OPENROUTER_API_KEY"]; EP="https://openrouter.ai/api/v1/embeddings"
MODELS=["google/gemini-embedding-2","nvidia/llama-nemotron-embed-vl-1b-v2:free"]

docs=[
 ("cell","The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration."),
 ("earnings","Quarterly revenue rose 23 percent to 4.2 billion dollars, driven by strong cloud sales."),
 ("baking","Preheat the oven to 220 degrees Celsius and bake the sourdough loaf for 35 minutes until golden."),
 ("treaty","The Treaty of Westphalia, signed in 1648, ended the Thirty Years War in Europe."),
 ("password","To reset your password, click the link in the verification email and choose a new one."),
 ("blackhole","A black hole forms when a massive star collapses under its own gravity at the end of its life."),
 ("lawsuit","The plaintiff alleges breach of contract and seeks damages of 1.5 million dollars."),
 ("amazon","The Amazon River discharges more water than the next seven largest rivers combined."),
]
queries=[
 ("cell","How do cells generate their energy?"),
 ("earnings","What were the company earnings and growth this quarter?"),
 ("baking","At what temperature and how long do you bake sourdough bread?"),
 ("treaty","Which agreement ended the Thirty Years War?"),
 ("password","Steps to change my account login password."),
 ("blackhole","How are black holes created from collapsing stars?"),
 ("lawsuit","A legal claim about a broken business contract and money owed."),
 ("amazon","Which river has the greatest water flow on Earth?"),
]

# find a TTF font
font=None
for p in glob.glob("/usr/share/fonts/**/*.ttf",recursive=True):
    if "DejaVuSans.ttf" in p or "LiberationSans" in p: font=p; break
def F(sz):
    try: return ImageFont.truetype(font,sz) if font else ImageFont.load_default(size=sz)
    except: return ImageFont.load_default(size=sz)

os.makedirs("textimg",exist_ok=True)
def render(key,text):
    img=Image.new("RGB",(1000,520),"white"); d=ImageDraw.Draw(img)
    d.text((40,30),"Document",font=F(26),fill=(120,120,120))
    y=110
    for line in textwrap.wrap(text,38):
        d.text((40,y),line,font=F(40),fill=(15,15,15)); y+=60
    path=f"textimg/{key}.png"; img.save(path); return path

S=requests.Session(); S.headers.update({"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
def emb(model, text=None, image=None):
    if text is not None: body={"model":model,"input":text,"encoding_format":"float"}
    else:
        b=base64.b64encode(open(image,"rb").read()).decode()
        body={"model":model,"input":[{"content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b}"}}]}],"encoding_format":"float"}
    for _ in range(4):
        r=S.post(EP,data=json.dumps(body),timeout=120)
        if r.status_code==200:
            v=np.array(r.json()["data"][0]["embedding"],dtype=np.float32); return v/np.linalg.norm(v)
        import time; time.sleep(3)
    raise RuntimeError(f"{model} {r.status_code} {r.text[:150]}")

paths=[render(k,t) for k,t in docs]
keys=[k for k,_ in docs]
# control: a real Flickr photo (should be far from text docs)
ctrl=sorted(glob.glob("images/*__m.jpg"))[0]

for model in MODELS:
    dimg=np.stack([emb(model,image=p) for p in paths])          # 8 document IMAGES
    qtxt=np.stack([emb(model,text=q) for _,q in queries])       # 8 paraphrase TEXT queries
    cv=emb(model,image=ctrl)
    sim=qtxt@dimg.T                                             # query x docimage
    r1=sum(int(np.argmax(sim[i])==i) for i in range(len(keys)))
    diag=np.mean([sim[i,i] for i in range(len(keys))])
    off=np.mean([sim[i,j] for i in range(len(keys)) for j in range(len(keys)) if i!=j])
    ctrl_sim=float(np.mean(qtxt@cv))
    print(f"\n=== {model} ===")
    print(f"  text-query -> document-IMAGE retrieval (paraphrased): recall@1 = {r1}/{len(keys)}")
    print(f"  mean cosine  matching pair = {diag:.3f}   non-matching = {off:.3f}   (gap {diag-off:+.3f})")
    print(f"  mean cosine  query-text vs a real photo (control)    = {ctrl_sim:.3f}")
    print("  per-query top match:")
    for i,(k,q) in enumerate(queries):
        j=int(np.argmax(sim[i])); print(f"    '{q[:42]:42}' -> {keys[j]:9} ({sim[i,j]:.3f}) {'OK' if j==i else 'WRONG (want %s)'%k}")

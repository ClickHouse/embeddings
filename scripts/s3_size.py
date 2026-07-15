#!/usr/bin/env python3
import sys, requests, collections, xml.etree.ElementTree as ET, urllib.parse
HOST="https://multimedia-commons.s3.us-west-2.amazonaws.com"
NS="{http://s3.amazonaws.com/doc/2006-03-01/}"
S=requests.Session()
def list_all(prefix):
    token=None
    while True:
        q={"list-type":"2","prefix":prefix,"max-keys":"1000"}
        if token: q["continuation-token"]=token
        r=S.get(HOST+"/?"+urllib.parse.urlencode(q),timeout=60); r.raise_for_status()
        root=ET.fromstring(r.content)
        for c in root.findall(NS+"Contents"):
            yield c.find(NS+"Key").text, int(c.find(NS+"Size").text)
        if root.find(NS+"IsTruncated").text!="true": break
        token=root.find(NS+"NextContinuationToken").text
def human(n):
    for u in ['B','KB','MB','GB','TB']:
        if n<1024 or u=='TB': return f"{n:,.1f} {u}"
        n/=1024
def report(prefix, depth):
    agg=collections.OrderedDict(); cnt=collections.Counter(); tot=0; nfiles=0
    for k,s in list_all(prefix):
        parts=k.split("/"); grp="/".join(parts[:depth]) if len(parts)>depth else k
        agg[grp]=agg.get(grp,0)+s; cnt[grp]+=1; tot+=s; nfiles+=1
    print(f"\n### {prefix}  -> {nfiles:,} files, {human(tot)}")
    for g,s in sorted(agg.items(),key=lambda x:-x[1]):
        print(f"   {human(s):>12}  {cnt[g]:>6} files  {g}")
    return tot
if __name__=="__main__":
    grand=0
    grand+=report("features/image/",3)
    grand+=report("features/keyframe/",3)
    grand+=report("features/audio/",3)
    grand+=report("tools/etc/",3)
    print(f"\n=== GRAND TOTAL (features + tools/etc): {human(grand)} ===")

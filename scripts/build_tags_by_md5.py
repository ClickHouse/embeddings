#!/usr/bin/env python3
# Build mmcommons.tags_by_md5: a fast, md5-sorted lookup of user_tags for the tag-cloud feature.
#
# yfcc_metadata is sorted by photo_id, so `md5 IN (...)` full-scans 100M rows (too slow for a live
# per-report lookup). This table is ORDER BY md5 so the lookup is a primary-key seek. Crucially its
# md5 is the SAME "stupidHex" form the emb_* tables (and hence topK search) return: the full 32-char
# hex md5 with each BYTE's leading zero dropped (per-byte, not whole-string). Applying that transform
# to yfcc_metadata.md5 reproduces the emb md5 exactly (verified: only ~0.65% of tagged photos are not
# in emb_siglip2 — the expected non-embedded gap). Only rows with non-empty user_tags are kept (~69M).
import os, http.client, ssl, time
PW = os.environ['CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD']
def q(sql, settings=''):
    c = http.client.HTTPSConnection('hvdvsqo23t.us-east-2.aws.clickhouse-staging.com', 8443,
                                    context=ssl.create_default_context(), timeout=1800)
    c.request('POST', '/?' + settings, body=sql.encode(),
              headers={'X-ClickHouse-User': 'default', 'X-ClickHouse-Key': PW})
    r = c.getresponse(); d = r.read().decode(); c.close()
    if r.status != 200: raise SystemExit(f"ERR {r.status}: {d[:400]}")
    return d.strip()

# per-byte leading-zero strip of the 32-char hex md5 -> emb "stupidHex" md5
STUPID = ("arrayStringConcat(arrayMap(i -> if(substring(md5, i*2-1, 1) = '0', "
          "substring(md5, i*2, 1), substring(md5, i*2-1, 2)), range(1, 17)))")
t0 = time.time()
q("DROP TABLE IF EXISTS mmcommons.tags_by_md5")
q("CREATE TABLE mmcommons.tags_by_md5 (md5 String, user_tags Array(String)) ENGINE = MergeTree ORDER BY md5")
q(f"""INSERT INTO mmcommons.tags_by_md5
      SELECT {STUPID} AS md5, user_tags
      FROM mmcommons.yfcc_metadata WHERE notEmpty(user_tags)""",
  'enable_parallel_replicas=1&max_parallel_replicas=3&max_memory_usage=48000000000&max_execution_time=1800')
n = q("SELECT count() FROM mmcommons.tags_by_md5")
miss = q("SELECT round(100*avg(md5 NOT IN (SELECT md5 FROM mmcommons.emb_siglip2)), 2) "
         "FROM (SELECT md5 FROM mmcommons.tags_by_md5 WHERE cityHash64(md5)%211=0)")
print(f"tags_by_md5: {n} rows, {miss}% not in emb_siglip2 (expect ~1% non-embedded), {time.time()-t0:.0f}s")

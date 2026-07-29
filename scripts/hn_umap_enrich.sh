#!/usr/bin/env bash
# Enrich the 18 HackerNews UMAP 2D tables (default.hackernews_umap_<model>_2d) with mapped
# integer coords mx,my,mz + a morton projection (proj_umap), mirroring mmcommons.umap_<ds>_2d,
# so the point-cloud tile / hover / overlay queries work unchanged for the HackerNews corpus.
# Also emits the per-model render constants the frontend needs:
#   2D: minx/maxx/miny/maxy/minc/maxc (mapping) + uq99 (z=0 per-pixel density for brightness)
#   3D: cx,cy,cz (centre), s (uniform scale), clo/chi (colour range), q99 (projected density)
# Idempotent: ADD COLUMN / ADD PROJECTION are IF NOT EXISTS; re-running only re-materialises.
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --query "$1" 2>&1 | grep -vi "unknown setting"; }
MODELS="arctic_m arctic_xs bge_base bge_large bge_small bow e5_base e5_large e5_small \
embeddinggemma granite_small gte_base gte_small jina_small jina_v3 kalm minilm nomic qwen3_8b"

echo "=== HN UMAP 2D enrich + constants  $(date +%T) ==="
for m in $MODELS; do
  T=default.hackernews_umap_${m}_2d
  T3=default.hackernews_umap_${m}_3d
  echo "--- $m  $(date +%T) ---"
  read minx maxx miny maxy minc maxc <<< "$(CL "SELECT min(x),max(x),min(y),max(y),min(color),max(color) FROM $T FORMAT TSV")"
  CL "ALTER TABLE $T
      ADD COLUMN IF NOT EXISTS mx UInt32 MATERIALIZED toUInt32(round(clamp((toFloat64(x)-($minx))/(($maxx)-($minx)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS my UInt32 MATERIALIZED toUInt32(round(clamp((toFloat64(y)-($miny))/(($maxy)-($miny)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS mz UInt16 MATERIALIZED toUInt16(round(clamp((toFloat64(color)-($minc))/(($maxc)-($minc)),0.,1.)*65535))"
  CL "ALTER TABLE $T MATERIALIZE COLUMN mx, MATERIALIZE COLUMN my, MATERIALIZE COLUMN mz SETTINGS mutations_sync=1"
  CL "ALTER TABLE $T ADD PROJECTION IF NOT EXISTS proj_umap (SELECT mx,my,mz,id ORDER BY mortonEncode(mx,my))"
  CL "ALTER TABLE $T MATERIALIZE PROJECTION proj_umap SETTINGS mutations_sync=1"
  uq99=$(CL "SELECT round(quantile(0.99)(cnt)) FROM (SELECT intDiv(mx,4194304) gx, intDiv(my,4194304) gy, count() cnt FROM $T GROUP BY gx,gy) FORMAT TSV")
  read cx cy cz s clo chi <<< "$(CL "SELECT round((min(x)+max(x))/2,4), round((min(y)+max(y))/2,4), round((min(z)+max(z))/2,4), round(greatest(max(x)-min(x),max(y)-min(y),max(z)-min(z))/2,4), round(min(color),4), round(max(color),4) FROM $T3 FORMAT TSV")"
  q99_3d=$(CL "WITH ((x-($cx))/($s)) AS nx, ((y-($cy))/($s)) AS ny SELECT round(quantile(0.99)(cnt)) FROM (SELECT intDiv(toUInt32(least(1.,greatest(0.,nx*0.35+0.5))*4294967295),4194304) gx, intDiv(toUInt32(least(1.,greatest(0.,ny*0.35+0.5))*4294967295),4194304) gy, count() cnt FROM $T3 GROUP BY gx,gy) FORMAT TSV")
  echo "CONST $m | 2d minx=$minx maxx=$maxx miny=$miny maxy=$maxy minc=$minc maxc=$maxc uq99=$uq99 | 3d cx=$cx cy=$cy cz=$cz s=$s clo=$clo chi=$chi q99=$q99_3d"
done
echo "=== HN UMAP ENRICH DONE  $(date +%T) ==="

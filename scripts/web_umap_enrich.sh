#!/usr/bin/env bash
# Enrich the 7 WonderfulWeb UMAP-2D tables (web_umap_{img,txt}_<model>_2d) with mapped integer coords
# mx,my,mz + a morton projection (proj_umap), and emit the per-model render constants the frontend needs,
# mirroring hn_umap_enrich.sh. Keyed by `url` (not id). Web has NO linear projection (emb tables carry only
# url+embedding+QBit), so projs are UMAP 2D + 3D only -> no `lq`.
#   2D: minx/maxx/miny/maxy/minc/maxc (mapping) + uq99 (z=0 per-pixel density for brightness)
#   3D: cx,cy,cz (centre), s (uniform scale), clo/chi (colour range), q99 (projected density)
# Idempotent (IF NOT EXISTS + re-materialise). Run AFTER the thumbnail build to avoid merge-pool contention.
set -uo pipefail
H=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
CL(){ /home/ubuntu/embeddings/clickhouse-new client --host "$H" --secure --user default \
      --password "$CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD" --query "$1" 2>&1 | grep -vi "unknown setting"; }
MODELS="img_clip img_nomic img_siglip2 txt_bge_large txt_e5_large txt_jina_v3 txt_nomic"

echo "=== WEB UMAP 2D enrich + constants  $(date +%T) ==="
for m in $MODELS; do
  T=default.web_umap_${m}_2d
  T3=default.web_umap_${m}_3d
  echo "--- $m  $(date +%T) ---"
  read minx maxx miny maxy minc maxc <<< "$(CL "SELECT min(x),max(x),min(y),max(y),min(color),max(color) FROM $T FORMAT TSV")"
  CL "ALTER TABLE $T
      ADD COLUMN IF NOT EXISTS mx UInt32 MATERIALIZED toUInt32(round(clamp((toFloat64(x)-($minx))/(($maxx)-($minx)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS my UInt32 MATERIALIZED toUInt32(round(clamp((toFloat64(y)-($miny))/(($maxy)-($miny)),0.,1.)*4294967295)),
      ADD COLUMN IF NOT EXISTS mz UInt16 MATERIALIZED toUInt16(round(clamp((toFloat64(color)-($minc))/(($maxc)-($minc)),0.,1.)*65535))"
  CL "ALTER TABLE $T MATERIALIZE COLUMN mx, MATERIALIZE COLUMN my, MATERIALIZE COLUMN mz SETTINGS mutations_sync=1"
  CL "ALTER TABLE $T ADD PROJECTION IF NOT EXISTS proj_umap (SELECT mx,my,mz,url ORDER BY mortonEncode(mx,my))"
  CL "ALTER TABLE $T MATERIALIZE PROJECTION proj_umap SETTINGS mutations_sync=1"
  uq99=$(CL "SELECT round(quantile(0.99)(cnt)) FROM (SELECT intDiv(mx,4194304) gx, intDiv(my,4194304) gy, count() cnt FROM $T GROUP BY gx,gy) FORMAT TSV")
  read cx cy cz s clo chi <<< "$(CL "SELECT round((min(x)+max(x))/2,4), round((min(y)+max(y))/2,4), round((min(z)+max(z))/2,4), round(greatest(max(x)-min(x),max(y)-min(y),max(z)-min(z))/2,4), round(min(color),4), round(max(color),4) FROM $T3 FORMAT TSV")"
  q99_3d=$(CL "WITH ((x-($cx))/($s)) AS nx, ((y-($cy))/($s)) AS ny SELECT round(quantile(0.99)(cnt)) FROM (SELECT intDiv(toUInt32(least(1.,greatest(0.,nx*0.35+0.5))*4294967295),4194304) gx, intDiv(toUInt32(least(1.,greatest(0.,ny*0.35+0.5))*4294967295),4194304) gy, count() cnt FROM $T3 GROUP BY gx,gy) FORMAT TSV")
  echo "CONST $m | 2d minx=$minx maxx=$maxx miny=$miny maxy=$maxy minc=$minc maxc=$maxc uq99=$uq99 | 3d cx=$cx cy=$cy cz=$cz s=$s clo=$clo chi=$chi q99=$q99_3d"
done
echo "=== WEB UMAP ENRICH DONE  $(date +%T) ==="

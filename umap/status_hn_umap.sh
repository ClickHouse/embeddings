#!/bin/bash
# Progress snapshot for the HN UMAP driver.
H2=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
ch() { curl -sS "https://${H2}:8443/" --user "default:${CH_PW:?set CH_PW}" --data-binary "$1"; }
cd /home/embed/gpu_embed
echo "=== $(date -u +%H:%M:%S)Z ==="
grep -E "PHASE|complete|ALL HN" umap/driver_hn_umap.log 2>/dev/null | tail -3
echo "-- reduce caches built (umap_work/*.meta) --"; ls umap_work/hn_*.meta 2>/dev/null | wc -l
echo "-- built umap tables (rows) --"
ch "SELECT name, formatReadableQuantity(total_rows) FROM system.tables WHERE name LIKE 'hackernews_umap_%' AND total_rows > 0 ORDER BY name FORMAT TSVRaw"
echo "-- running jobs --"; ps -ef | grep hn_umap_atlas | grep -v grep | grep -oE 'model [a-z0-9_]+ (--reduce-only|--dims [23]d)' | sort | uniq -c
echo "-- gpu util --"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | paste -sd' | '

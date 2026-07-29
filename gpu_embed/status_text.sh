#!/bin/bash
# Compact progress + throughput snapshot for the text-embedding driver.
H2=hvdvsqo23t.us-east-2.aws.clickhouse-staging.com
ch() { curl -sS "https://${H2}:8443/" --user "default:${CH_PW:?set CH_PW}" --data-binary "$1"; }
cd /home/embed/gpu_embed
TOTAL=37799159
DRV=$(ls -t driver_text*.log 2>/dev/null | head -1)
CUR=$(grep -oE '########## [a-z0-9_]+ : starting' "$DRV" 2>/dev/null | tail -1 | awk '{print $2}')
echo "=== $(date -u +%H:%M:%S)Z  current model: ${CUR:-?} ==="
echo "-- built tables (rows) --"
ch "SELECT name, formatReadableQuantity(total_rows) FROM system.tables WHERE name LIKE 'hackernews_embeddings_%' AND name NOT LIKE '%qwen3%' ORDER BY name FORMAT TSVRaw"
if [ -n "$CUR" ]; then
  STG=$(ch "SELECT count() FROM stage_hn_${CUR}" 2>/dev/null || echo 0)
  [[ "$STG" =~ ^[0-9]+$ ]] || STG=0
  echo "-- $CUR staged: ${STG} / ${TOTAL} ($(awk "BEGIN{printf \"%.1f\", ${STG}*100/$TOTAL}")%) --"
  agg=0
  for g in 0 1 2 3; do
    line=$(grep "rows/s" text_${CUR}_gpu${g}.log 2>/dev/null | tail -1)
    r=$(echo "$line" | grep -oE '[0-9]+ rows/s' | grep -oE '[0-9]+')
    prog=$(echo "$line" | grep -oE '[0-9]+/[0-9]+ chunks' | head -1)
    if grep -qE "DONE|worker finished ok" text_${CUR}_gpu${g}.log 2>/dev/null; then st=DONE; else st="${r:-0} r/s"; fi
    echo "   gpu$g: ${prog:-loading}  ${st}"
    agg=$((agg + ${r:-0}))
  done
  if [ "$agg" -gt 0 ]; then
    echo "   aggregate: ${agg} rows/s  (~$(awk "BEGIN{printf \"%.0f\", ($TOTAL-${STG})/${agg}/60}") min left this model)"
  else
    echo "   aggregate: 0 rows/s (loading / finalizing)"
  fi
fi
echo "-- gpu util: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | paste -sd' ') --"

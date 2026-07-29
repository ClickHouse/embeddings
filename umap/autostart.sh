#!/bin/bash
# AUTO-CONTINUE: when siglip2 finishes AND venv_umap (RAPIDS) is ready, run a small validation of the
# UMAP pipeline; ONLY if it passes, launch the full atlas build. Safe to leave running unattended.
# Disarm:  pkill -f umap/autostart.sh   (before siglip2 completes)
# Watch:   tail -f umap/autostart.log
set -u
cd "$(dirname "$0")/.."
LOG=umap/autostart.log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

export CH_HOST=$(grep -aoE 'CH_HOST=[^ ]+' ~/.bash_history | tail -1 | sed 's/CH_HOST=//')
export CH_PW=$(grep -aoE "CH_PW='[^']*'|CH_PW=[^ ;]+" ~/.bash_history | tail -1 | sed -E "s/CH_PW=//; s/^'//; s/'\$//")
export CH_USER=default
ch(){ curl -sS "https://${CH_HOST}:8443/" --user "default:${CH_PW}" --data-binary "$1"; }

say "armed. waiting for siglip2 to finish ..."
while :; do
  grep -q 'ALL LIGHT MODELS COMPLETE' run_seq.out 2>/dev/null && break
  { [ "$(ls emb/siglip2/*.done 2>/dev/null | wc -l)" -ge 4096 ] && ! pgrep -f 'models siglip2' >/dev/null; } && break
  sleep 60
done
say "siglip2 complete. waiting for venv_umap (RAPIDS) ..."
while :; do venv_umap/bin/python -c 'import cuml,cupy,clickhouse_connect' 2>/dev/null && break; sleep 30; done

say "validation: umap_atlas on nomic, --limit 200000 (gpu0) ..."
CUDA_VISIBLE_DEVICES=0 venv_umap/bin/python umap/umap_atlas.py \
    --model nomic --limit 200000 --fit-sample 100000 --pca-sample 200000 >> "$LOG" 2>&1
rc=$?
n2=$(ch "SELECT count() FROM mmcommons.umap_nomic_2d"); n3=$(ch "SELECT count() FROM mmcommons.umap_nomic_3d")
say "validation rc=$rc  umap_nomic_2d=$n2  umap_nomic_3d=$n3"
# always clear the small validation output + nomic reduced cache so the full run is clean
ch "TRUNCATE TABLE mmcommons.umap_nomic_2d" >/dev/null; ch "TRUNCATE TABLE mmcommons.umap_nomic_3d" >/dev/null
rm -f umap_work/nomic_pca*.f32 umap_work/nomic_pca*.md5 umap_work/nomic_pca*.meta
if [ "$rc" -ne 0 ] || [ "${n2:-0}" -lt 100000 ] || [ "${n3:-0}" -lt 100000 ]; then
  say "VALIDATION FAILED -> NOT launching full run. Inspect $LOG."
  exit 1
fi
say "validation OK. launching FULL umap (all 3 models, 2d+3d) ..."
bash umap/run_umap.sh >> "$LOG" 2>&1
say "FULL UMAP finished (rc=$?). Tables: mmcommons.umap_{nomic,clip,siglip2}_{2d,3d}"

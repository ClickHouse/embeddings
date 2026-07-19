#!/bin/bash
# Finish siglip2's contiguous tail WITHOUT the rotation tail-collision (all GPUs redoing the same
# top prefixes ~4x). Partition the UNDONE prefixes into NGPU DISJOINT slices so each GPU embeds unique
# prefixes. Supervised (auto-relaunch on crash). autostart.sh fires on completion (4096 done, no workers).
set -e
cd "$(dirname "$0")"
: "${CH_HOST:?export CH_HOST}"; : "${CH_PW:?export CH_PW}"; export CH_HOST CH_PW CH_USER
PY="$PWD/venv/bin/python"; OUT="$PWD/emb"; NGPU=$(nvidia-smi -L | wc -l)
"$PY" -c "print('\n'.join('%03x'%i for i in range(4096)))" | sort > /tmp/sig_all.txt
ls "$OUT/siglip2"/*.done 2>/dev/null | sed 's#.*/##;s/\.done$//' | sort > /tmp/sig_done.txt
comm -23 /tmp/sig_all.txt /tmp/sig_done.txt > /tmp/sig_undone.txt
echo "undone: $(wc -l < /tmp/sig_undone.txt) prefixes ($(head -1 /tmp/sig_undone.txt)..$(tail -1 /tmp/sig_undone.txt))"
for g in $(seq 0 $((NGPU-1))); do awk "NR % $NGPU == $g" /tmp/sig_undone.txt > "finish_sig.$g.txt"; echo "  gpu$g: $(wc -l < finish_sig.$g.txt) unique prefixes"; done
sup(){ local g=$1 rc; set +e; : > "gpu$g.log"; while :; do
    CUDA_VISIBLE_DEVICES=$g "$PY" embed_images.py --prefix-file "finish_sig.$g.txt" --out "$OUT" --models siglip2 --batch 256 >> "gpu$g.log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && { echo "[sup $(date +%T)] gpu$g done rc=0" >> "gpu$g.log"; break; }
    [ "$(ls "$OUT"/siglip2/*.done 2>/dev/null | wc -l)" -ge 4096 ] && break
    echo "[sup $(date +%T)] gpu$g rc=$rc; relaunch in 10s" >> "gpu$g.log"; sleep 10
  done; }
for g in $(seq 0 $((NGPU-1))); do sup "$g" & done
wait
echo "siglip2 tail COMPLETE $(date '+%F %T')"

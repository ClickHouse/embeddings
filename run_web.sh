#!/bin/bash
# Launch the WonderfulWeb embedding pipelines: image (siglip2/clip/nomic) + text (bge_large/e5_large/
# jina_v3/nomic), one worker per GPU each (image + text share each GPU; image is read-bound so text uses
# the spare GPU cycles). Supervised (relaunch on crash) + resumable (per-shard .done markers).
# Usage: WW_HOST/WW_PW + CH_HOST/CH_PW in env, then: ./run_web.sh [image|text|both]
set -u
: "${WW_HOST:?}"; : "${WW_PW:?}"; : "${CH_HOST:?}"; : "${CH_PW:?}"
export WW_HOST WW_PW CH_HOST CH_PW CH_USER HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$(dirname "$0")"
PY="$PWD/venv/bin/python"
NGPU=$(nvidia-smi -L | wc -l)
WHICH="${1:-both}"
MAXTRIES=${MAXTRIES:-1000}

supervise(){   # $1=kind(img|txt) $2=gpu $3=script $4=extra
  set +e
  local kind=$1 gpu=$2 script=$3
  local log="web_${kind}_gpu${gpu}.log" tries=0
  while :; do
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$script" --gpu "$gpu" --ngpu "$NGPU" $4 >>"$log" 2>&1
    rc=$?; [ $rc -eq 0 ] && { echo "[$kind gpu$gpu] finished ok" >>"$log"; break; }
    tries=$((tries+1)); [ $tries -ge $MAXTRIES ] && { echo "[$kind gpu$gpu] gave up after $tries" >>"$log"; break; }
    echo "[$kind gpu$gpu] exited rc=$rc, relaunch #$tries in 15s" >>"$log"; sleep 15
  done
}

pids=()
for g in $(seq 0 $((NGPU-1))); do
  if [ "$WHICH" = image ] || [ "$WHICH" = both ]; then
    : > "web_img_gpu${g}.log"; supervise img "$g" embed_web_images.py "--batch 64" & pids+=($!)
  fi
  if [ "$WHICH" = text ] || [ "$WHICH" = both ]; then
    : > "web_txt_gpu${g}.log"; supervise txt "$g" embed_web_text.py "--batch 128" & pids+=($!)
  fi
done
echo "launched ${#pids[@]} workers ($WHICH), $NGPU GPUs. monitor: tail -f web_img_gpu0.log web_txt_gpu0.log"
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL WEB WORKERS DONE"

#!/bin/bash
# Build default.web_thumbs (64x64 gamma-correct RGB) from default.web_screenshots via streaming range pulls.
# Server-side SQL resize is impossible (arrayMap replicates the 4.9M-elem screenshot -> 56 GiB), and the
# dest->this-box link is slow single-stream (~31 MiB/s) but AGGREGATES across parallel streams (~155 MiB/s @ 8).
# So: split the url PK into K ranges, pull each as a compressed RowBinary STREAM (no LIMIT/ORDER BY -> no
# lazy-mat/sort OOM), pipe to resize_stream.py, P ranges concurrent. Resumable (per-range markers), retried.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
H="https://hvdvsqo23t.us-east-2.aws.clickhouse-staging.com:8443"
DP="${CLICKHOUSE_CLOUD_EMBEDDINGS_PASSWORD:-}"
K=1024; P=32
STATE=/tmp/webthumbs; mkdir -p "$STATE"; LOG=/tmp/web_thumbs.log; BF="$STATE/boundaries"
[ -z "$DP" ] && { echo "$(date +%T) ABORT: dest pw empty" >>"$LOG"; exit 1; }
CL(){ curl -s --max-time "${2:-120}" "$H/?user=default&password=$DP" --data-binary "$1"; }

if [ ! -s "$BF" ]; then
  total=$(CL "SELECT count() FROM default.web_screenshots FORMAT TSV" 60 | tr -d '[:space:]')
  step=$(( total / K )); [ "$step" -lt 1 ] && step=1
  echo "$(date +%T) computing $K boundaries (total=$total step=$step)" >>"$LOG"
  CL "SELECT any(url) FROM (SELECT url, intDiv(rowNumberInAllBlocks(),$step) AS bk FROM (SELECT url FROM default.web_screenshots ORDER BY url)) GROUP BY bk ORDER BY bk FORMAT TSV" 180 > "$BF"
fi
NB=$(wc -l < "$BF")
echo "=== web_thumbs_stream start $(date) boundaries=$NB P=$P ===" >>"$LOG"

do_range(){
  set -o pipefail
  local i=$1 m="$STATE/done_$1"
  [ -f "$m" ] && return 0
  mapfile -t B < "$BF"; local n=${#B[@]}
  local lo="" hi=""
  (( i > 0 ))      && lo="${B[$i]}"
  (( i + 1 < n ))  && hi="${B[$((i+1))]}"
  local elo=${lo//\\/\\\\}; elo=${elo//\'/\\\'}
  local ehi=${hi//\\/\\\\}; ehi=${ehi//\'/\\\'}
  local hicond=""; [ -n "$hi" ] && hicond=" AND url < '$ehi'"
  local a
  for a in 1 2 3; do
    if curl -s --compressed -H "Accept-Encoding: zstd" --max-time 10800 \
         "$H/?user=default&password=$DP&enable_http_compression=1&max_threads=3&max_memory_usage=8000000000" --data-binary \
         "SELECT url, screenshot FROM default.web_screenshots WHERE url >= '$elo'$hicond FORMAT RowBinary" \
         | python3 "$DIR/resize_stream.py" "r$i" 2>>"${LOG}.err"; then
      touch "$m"; echo "$(date +%T) range $i OK (attempt $a)" >>"$LOG"; return 0
    fi
    echo "$(date +%T) range $i FAIL attempt $a" >>"$LOG"; sleep 10
  done
  echo "$(date +%T) range $i GAVEUP" >>"$LOG"; return 1
}
export -f do_range; export H DP STATE LOG BF DIR

seq 0 $(( NB - 1 )) | xargs -P "$P" -I{} bash -c 'do_range "$1"' _ {}
echo "=== web_thumbs_stream ALL DONE $(date) ===" >>"$LOG"

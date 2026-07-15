#!/bin/bash
# $1=photo_id $2=size $3=url $4=outpath ; saves image, skips if already present
[ -s "$4" ] && exit 0
if curl -s -A "size-survey/1.0" -H "Accept-Encoding: identity" --max-time 180 --retry 6 --retry-delay 2 -o "$4.tmp" "$3"; then
  mv "$4.tmp" "$4"
else
  rm -f "$4.tmp"
fi

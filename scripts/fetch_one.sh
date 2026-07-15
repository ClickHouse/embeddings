#!/bin/bash
# args: $1=photo  $2=size  $3=url   ; appends "photo<TAB>size<TAB>code<TAB>bytes" to $OUT
out=$(curl -s -o /dev/null -A "size-survey/1.0" -H "Accept-Encoding: identity" \
      --max-time 180 --retry 6 --retry-delay 2 -w '%{http_code} %{size_download}' "$3")
printf '%s\t%s\t%s\t%s\n' "$1" "$2" "${out% *}" "${out#* }" >> "$OUT"

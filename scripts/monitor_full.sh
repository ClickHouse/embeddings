cd /home/ubuntu/embeddings
for i in $(seq 1 170); do
  if grep -q ALL_SHARDS_PROCESSED drive_full.out 2>/dev/null; then break; fi
  sleep 300
done
echo "DRIVER FINISHED"
echo "done=$(grep -c '^DONE' drive_full.out) fail=$(grep -c '^FAIL' drive_full.out) outfiles=$(ls results/qwen_full/ 2>/dev/null|wc -l)/512"
grep '^FAIL' drive_full.out 2>/dev/null | head

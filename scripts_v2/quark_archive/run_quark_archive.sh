#!/bin/bash
# Keep the Quark archive running until it finishes (resumable state makes restarts cheap).
cd /root
LOG=/root/autodl-tmp/redl_work/quark_archive.log
for i in $(seq 1 200); do
  echo "[supervisor] attempt $i $(date +%H:%M)" >> "$LOG"
  /root/miniconda3/bin/python /root/quark_archive.py --uploaders 12 --converters 4 --part-mb 32 --skip-png >> "$LOG" 2>&1
  rc=$?
  echo "[supervisor] exited rc=$rc $(date +%H:%M)" >> "$LOG"
  # stop if everything is done
  if grep -q "ALL DONE" "$LOG"; then
    echo "[supervisor] all done, exiting" >> "$LOG"
    break
  fi
  sleep 20
done

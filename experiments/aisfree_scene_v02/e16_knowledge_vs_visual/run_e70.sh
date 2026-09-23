#!/bin/bash
cd "E:/临时会话/visual_reliable_baseline" || exit 1
PY=/d/Program_files/anaconda3/python.exe
for i in 1 2 3; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 E70_SUBS=20000 E70_NINNER=8 \
    "$PY" -X utf8 -u e70_gate_share.py > e70_out.txt 2>&1
  if grep -q '配对 gate_share' e70_out.txt; then echo "OK e70"; break; fi
  echo "attempt $i: $(grep -vE 'OMP|Hint|Warning|warn' e70_out.txt | tail -1 | head -c 140)"
done
grep -vE 'OMP|Hint|Warning|warn|^$' e70_out.txt | tail -14
echo E70DONE

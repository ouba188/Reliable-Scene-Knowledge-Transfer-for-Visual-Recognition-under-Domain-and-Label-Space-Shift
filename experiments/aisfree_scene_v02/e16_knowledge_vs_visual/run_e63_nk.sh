#!/bin/bash
# ①′: inner=3, gate WITHOUT the raw knowledge vector (gate_nk arm), same folds as ① for a clean comparison.
cd "E:/临时会话/visual_reliable_baseline" || exit 1
PY=/d/Program_files/anaconda3/python.exe
# self-match guard: build the pattern by concatenation so this checker's own cmdline never contains it
busy() { "$PY" -c "
import psutil,sys
pat='e63_gate'+'_knowledge.py'
sys.exit(0 if any(pat in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])) else 1)
"; }
while busy; do sleep 60; done
echo "=== ①′: inner=3, gate 去掉原始 71 维知识 (gate_nk) ==="
for i in 1 2 3; do
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 E63_SUBS=20000 E63_NINNER=3 E63_TAG=n3nk \
    "$PY" -X utf8 -u e63_gate_knowledge.py > e63_out_n3_nk.txt 2>&1
  if grep -q '配对 gate_nk' e63_out_n3_nk.txt; then echo "OK inner=3 nk"; break; fi
  echo "attempt $i: $(grep -vE 'OMP|Hint|Warning|warn' e63_out_n3_nk.txt | tail -1 | head -c 130)"
done
grep -vE 'OMP|Hint|Warning|warn|^$' e63_out_n3_nk.txt | tail -16
echo NKDONE

#!/bin/bash
cd "E:/临时会话/visual_reliable_baseline" || exit 1
PY=/d/Program_files/anaconda3/python.exe
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "=== ① VV-only 特征（已知池 + 未知池）==="
"$PY" -X utf8 -u e81_vv_features.py both > e81_out.txt 2>&1
if ! grep -q 'DONE unknown' e81_out.txt; then
  echo "e81 失败:"; grep -vE 'OMP|Hint|Warning|warn' e81_out.txt | tail -5; exit 1
fi
grep -vE 'OMP|Hint|Warning|warn|^$' e81_out.txt | tail -6

echo
echo "=== ② 误吸收测量（NAR 视觉半边）==="
"$PY" -X utf8 -u e82_absorption.py > e82_out.txt 2>&1
grep -vE 'OMP|Hint|Warning|warn|^$' e82_out.txt | tail -24
echo NARDONE

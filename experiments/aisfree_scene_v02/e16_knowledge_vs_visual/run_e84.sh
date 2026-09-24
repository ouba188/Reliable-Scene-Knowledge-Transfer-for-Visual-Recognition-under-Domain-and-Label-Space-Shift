#!/bin/bash
cd "E:/临时会话/visual_reliable_baseline" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY="D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe"
"$PY" -X utf8 -u e84_build_224.py 1000 > e84_out.txt 2>&1
grep -vE 'Warning|warn|^$' e84_out.txt | tail -18
echo E84DONE

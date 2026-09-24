#!/bin/bash
cd "E:/临时会话/visual_reliable_baseline" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
/d/Program_files/anaconda3/python.exe -X utf8 -u e88_nar_knowledge.py both > e88_out.txt 2>&1
grep -vE 'OMP|Hint|Warning|warn|^$' e88_out.txt | tail -30
echo E88DONE

#!/bin/bash
# 24-port chain: AIS dims -> features (3 FOV variants) -> the arms table.
cd "E:/临时会话/visual_reliable_baseline" || exit 1
PY=/d/Program_files/anaconda3/python.exe
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 DS_DIR=dataset_all FEAT_DIR=features_all AIS_DIM=ais_dim_all.npy AIS_OUT=ais_dim_all.npy

echo "=== 1) AIS 登记长度（71,801 片） ==="
"$PY" -X utf8 -u e103_ais_dims_local.py > e103_all_out.txt 2>&1
grep -vE 'OMP|Hint|Warning|warn|^$' e103_all_out.txt | tail -6

echo
echo "=== 2) 特征（c64 / c112 / full224 三种视场） ==="
"$PY" -X utf8 -u e106_fov.py extract > e106_all_out.txt 2>&1
grep -vE 'OMP|Hint|Warning|warn|^$' e106_all_out.txt | tail -6

echo
echo "=== 3) 24 港三臂对照 ==="
"$PY" -X utf8 -u e107_stack.py > e107_all_out.txt 2>&1
grep -vE 'OMP|Hint|Warning|warn|^$' e107_all_out.txt | tail -18

echo ALLDONE

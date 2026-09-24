#!/bin/bash
# Chain for when the Quark download lands: 24-port build -> features -> the three arms -> tables.
# Everything here is already validated piecewise (e109 dry run, e106 FOV, e103/e104 AIS length, e107 stacking);
# this only wires them together so the run needs no supervision.
cd "E:/临时会话/visual_reliable_baseline" || exit 1
PY=/d/Program_files/anaconda3/python.exe
CLEARPY="D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 DS_DIR=dataset_all FEAT_DIR=features_all

echo "=== 0) 下载是否齐了 ==="
n_done=$(ls /e/tif_local_done/*.tif 2>/dev/null | wc -l)
n_part=$(ls /e/tif_local_done/*.qkdownloading 2>/dev/null | wc -l)
echo "  已完成 $n_done / 430，仍在下载 $n_part"
if [ "$n_part" -gt 0 ]; then echo "  仍有文件在下载 —— 提示：构建会自动跳过 .qkdownloading，可先跑已到部分"; fi

echo
echo "=== 1) 24 港裁剪构建（可续跑） ==="
OMP_NUM_THREADS=4 "$CLEARPY" -X utf8 -u e109_build_all.py 800 > e109_out.txt 2>&1
tail -6 e109_out.txt
"$PY" -X utf8 -c "
import csv,collections
rows=list(csv.DictReader(open(r'E:/临时会话/visual_reliable_baseline/dataset_all/index.csv',encoding='utf-8')))
print('  数据集: %d 片 / %d 港' % (len(rows), len(set(r['port'] for r in rows))))
print('  按港:', dict(collections.Counter(r['port'] for r in rows).most_common()))
"

echo
echo "=== 2) 提特征（c64 视场 + 全视场，用于对照） ==="
"$PY" -X utf8 -u e106_fov.py extract > e106x_out.txt 2>&1 || tail -3 e106x_out.txt

echo
echo "=== 3) 三臂对照（c64 / +AIS 长度 / +知识） ==="
"$PY" -X utf8 -u e107_stack.py > e107x_out.txt 2>&1
grep -vE 'OMP|Hint|Warning|warn|^$' e107x_out.txt | tail -16

echo
echo "（跳过 e82：其输入是独立的 V5 未知类片，与新池无关）"

echo ALLDONE

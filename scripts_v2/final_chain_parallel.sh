#!/bin/bash
# 最终链（并行版）：3 个分片并行跑对象表 -> Newcastle -> 合并 -> 提取命中要素 -> 定向定年 -> join -> 汇总
cd /e/Hermes || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/d/Program_files/anaconda3/python.exe
KS="E:/临时会话/knowledge_set_841"
LOG=scripts/out/final_chain.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

# 分片（按产品序号取模，保证每片的港口分布均衡）
$PY -X utf8 - <<'PYEOF' >> "$LOG" 2>&1
names = open('scripts/out/objtable_products.txt', encoding='utf-8').read().split()
for i in range(3):
    part = names[i::3]
    open('scripts/out/objtable_part%d.txt' % i, 'w', newline='').write(' '.join(part))
    print('分片 %d: %d 产品' % (i, len(part)))
PYEOF

say "1/7 并行跑对象表（3 分片）"
for i in 0 1 2; do
    $PY -X utf8 scripts/build_object_table_local.py --products $(cat scripts/out/objtable_part$i.txt) \
        --out "$KS/objects/objects_part$i.csv.gz" >> "$LOG" 2>&1 &
done
wait

say "2/7 Newcastle 12 产品"
$PY -X utf8 scripts/build_object_table_local.py --products $(cat scripts/out/objtable_newcastle.txt) \
    --out "$KS/objects/objects_newcastle.csv.gz" >> "$LOG" 2>&1

say "3/7 合并 + 分港口汇总"
$PY -X utf8 scripts/merge_object_tables.py "$KS/objects/objects_all.csv.gz" \
    "$KS/objects/objects_part0.csv.gz" "$KS/objects/objects_part1.csv.gz" "$KS/objects/objects_part2.csv.gz" \
    "$KS/objects/objects_newcastle.csv.gz" >> "$LOG" 2>&1

say "4/7 提取命中要素（element:id）"
$PY -X utf8 -c "
import csv, gzip
need = set()
with gzip.open('$KS/objects/objects_all.csv.gz','rt',encoding='utf-8-sig',newline='') as fh:
    for row in csv.DictReader(fh):
        e, i = row.get('facility_osm_element',''), row.get('facility_osm_id','')
        if i: need.add((e or 'way') + ':' + i)
open('scripts/out/need_dates2.txt','w',newline='').write('\n'.join(sorted(need)))
print('命中要素唯一数:', len(need))
" >> "$LOG" 2>&1

say "5/7 定向定年"
$PY -X utf8 scripts/date_facility_ways.py --ids-file scripts/out/need_dates2.txt >> "$LOG" 2>&1

say "6/7 回填设施定年 + AIS 类别层"
$PY -X utf8 scripts/join_facility_dates.py "$KS/objects/objects_all.csv.gz" \
    --out "$KS/objects/objects_dated.csv.gz" >> "$LOG" 2>&1
$PY -X utf8 scripts/join_ais_class.py "$KS/objects/objects_dated.csv.gz" \
    --out "$KS/objects/objects_classed.csv.gz" >> "$LOG" 2>&1

say "7/7 最终汇总"
$PY -X utf8 scripts/merge_object_tables.py "$KS/objects/objects_final.csv.gz" \
    "$KS/objects/objects_classed.csv.gz" >> "$LOG" 2>&1

say "DONE"

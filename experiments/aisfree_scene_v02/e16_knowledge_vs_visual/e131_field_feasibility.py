"""e131: feasibility of a SAR-only port-wide vessel field (option 1), measured before proposing it.

The AIS block's value (E18t) is coverage: a port-wide gridded field of density/queue/heading built from AIS. The legal
substitute would aggregate SAR detections ACROSS acquisitions onto a port grid and recompute the same eleven quantities from
that field. Before building it, measure whether the field would even be populated:

  per port: products in the object table, total detections, the UTM extent, and the detection counts per 500 m cell once all
  that port's products are pooled; plus whether the heading field (obb_long_deg) is populated.
A cell that holds 0-1 detections across the whole archive cannot support a queue/density feature.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
CELL = 500.0            # the AIS traffic field's resolution is sub-km per cell


def main():
    per = defaultdict(lambda: {'n': 0, 'prod': set(), 'x': [], 'y': [], 'head': 0, 'cls': defaultdict(int)})
    with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='replace') as f:
        for r in csv.DictReader(f):
            p = r.get('port') or '?'
            d = per[p]
            d['n'] += 1
            d['prod'].add(r.get('product_id') or '')
            try:
                d['x'].append(float(r['world_x'])); d['y'].append(float(r['world_y']))
            except (KeyError, ValueError):
                pass
            if (r.get('obb_long_deg') or '').strip() not in ('', 'nan'):
                d['head'] += 1
            c = (r.get('fine_class') or r.get('ais_final_class') or '?').strip()
            if c:
                d['cls'][c] += 1
    print('%-18s %7s %7s %11s %11s %11s %9s %10s' % ('port', '对象', '产品', 'x 跨度km', 'y 跨度km', '中位cells', '有朝向%', '类数'))
    rows = []
    for p, d in sorted(per.items(), key=lambda kv: -kv[1]['n']):
        if len(d['x']) < 50:
            continue
        x = np.array(d['x']); y = np.array(d['y'])
        cx = np.floor((x - x.min()) / CELL).astype(int); cy = np.floor((y - y.min()) / CELL).astype(int)
        cell = cx.astype(np.int64) * 100000 + cy
        _, cnt = np.unique(cell, return_counts=True)
        occ = cnt.mean()
        rows.append((p, d['n'], len(d['prod']), (x.max() - x.min()) / 1000, (y.max() - y.min()) / 1000,
                     occ, 100.0 * d['head'] / max(1, d['n']), len(d['cls'])))
        print('%-18s %7d %7d %11.1f %11.1f %11.1f %8.1f%% %10d'
              % (p, d['n'], len(d['prod']), rows[-1][3], rows[-1][4], occ, rows[-1][6], rows[-1][7]))
    print('')
    occ = np.array([r[5] for r in rows])
    print('各港被占用 cell 的平均检测数：中位 %.1f ｜ p10 %.1f ｜ 最小 %.1f' % (np.median(occ), np.percentile(occ, 10), occ.min()))
    print('朝向字段填充率：中位 %.1f%% ｜ 最小 %.1f%%' % (np.median([r[6] for r in rows]), min(r[6] for r in rows)))
    print('')
    print('⇒ %s' % ('可建 SAR 全港场 ✓（cell 内有足够检测 ✓）' if np.median(occ) >= 20
                    else '场会很稀疏 ⚠（cell 平均检测数 %.1f ⇒ 需更大 cell 或更长窗口 ✓）' % np.median(occ)))


main()

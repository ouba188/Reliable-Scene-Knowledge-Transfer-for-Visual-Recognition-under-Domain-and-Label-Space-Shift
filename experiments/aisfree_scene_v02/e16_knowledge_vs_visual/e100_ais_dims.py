"""e100: AIS-reported vessel dimensions -- the same 'size' axis, read from a reliable source.

e99 rejected size when measured from SAR detection boxes (obb_long_px tracks the bright scatterers, ~10% occupancy,
not the hull): it gave a known-class BA of 0.1988 vs 0.2070 and a novelty AUC of exactly 0.500, with identical
median lengths for known and unknown. The physical claim was never tested. The AIS lookup table on disk carries the
registered dimensions per MMSI (length, width, dwt), which is a physical measurement independent of the type code --
only length/width/dwt are used here, never ais_type_code, so no label leaks in.

Steps: chip -> MMSI via geo/<port>.geojson -> dimensions from mmsi_sarais_lookup.csv -> (a) sanity: do the classes
separate in length the way real ships do? (b) the open-set question: do the unknown types occupy a different length
distribution from the known ones?
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
DS = ROOT / 'dataset244'
GEOJ = KS / 'geo'
LOOK = KS / 'mmsi/mmsi_sarais_lookup.csv'
OUT = ROOT / 'features_244'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)

# MMSI per chip from the geo layer (textual join, unaffected by the coordinate offset problem)
mmsi_of = {}
for pj in sorted(set(ports.tolist())):
    f = GEOJ / ('%s.geojson' % pj)
    if not f.is_file():
        continue
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        pr = ft['properties']
        k = (pj, pr.get('product'), pr.get('pol'), pr.get('det'))
        m = (pr.get('mmsi') or '').strip()
        if m and m != '0':
            mmsi_of[k] = m
print('geo 中的 MMSI 条目 %d' % len(mmsi_of), flush=True)

dim = {}
with LOOK.open(encoding='utf-8-sig') as fh:
    for row in csv.DictReader(fh):
        m = (row.get('mmsi') or '').strip()
        if not m:
            continue
        try:
            L, W = float(row.get('length') or 0), float(row.get('width') or 0)
            D = float(row.get('dwt') or 0)
        except ValueError:
            continue
        if L > 0:
            dim[m] = (L, W, D)
print('尺寸表 %d 个 MMSI' % len(dim), flush=True)

L = np.full(len(idx), np.nan)
cov = 0
for i, r in enumerate(idx):
    m = mmsi_of.get((r['port'], r['product'], r['pol'], r['det']))
    if m and m in dim:
        L[i] = dim[m][0]
        cov += 1
print('芯片尺寸覆盖 %d / %d (%.0f%%)' % (cov, len(idx), 100 * cov / len(idx)), flush=True)
np.save(OUT / 'ais_length_244.npy', L)

print('')
print('自检：各类的 AIS 登记长度（应有船型学上的区分：拖轮短、散货/集装箱长）')
print('%-24s %6s %8s %8s %8s' % ('class', 'n', '中位长(m)', 'p25', 'p75'))
for c in sorted(set(cls.tolist()), key=lambda x: -int((cls == x).sum())):
    m = (cls == c) & np.isfinite(L)
    if m.sum() < 10:
        continue
    v = L[m]
    print('%-24s %6d %8.0f %8.0f %8.0f' % (c, int(m.sum()), np.median(v), np.percentile(v, 25), np.percentile(v, 75)))

print('')
print('未知类 vs 已知类的长度分布:')
kn = L[known & np.isfinite(L)]
un = L[(~known) & np.isfinite(L)]
print('  已知 n=%d 中位 %.0f m (p10 %.0f, p90 %.0f)' % (len(kn), np.median(kn), np.percentile(kn, 10), np.percentile(kn, 90)))
print('  未知 n=%d 中位 %.0f m (p10 %.0f, p90 %.0f)' % (len(un), np.median(un), np.percentile(un, 10), np.percentile(un, 90)))
lab = np.r_[np.zeros(len(kn)), np.ones(len(un))]
print('★ 长度对"已知 vs 未知"的 AUC: %.3f   （视觉 OOD 0.493 / 知识侧 0.509 / SAR 检测框 0.500）' % (
    roc_auc_score(lab, np.r_[kn, un])))
print('  逐未知类中位长度（越大越可能被长度识别为"不寻常"）:')
for c in sorted(set(cls[~known].tolist())):
    m = (cls == c) & np.isfinite(L)
    if m.sum() >= 10:
        print('    %-24s n=%5d 中位 %6.0f m' % (c, int(m.sum()), np.median(L[m])))

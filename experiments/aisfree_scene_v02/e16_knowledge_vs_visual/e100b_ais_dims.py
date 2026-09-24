"""e100b: AIS dimensions, joined the RIGHT way -- per-detection unique spatial matches from the object table.

e100 used geo/<port>.geojson's mmsi field, which is an unfiltered spatial association: coverage 1 % and the
remaining links wrong (a 'chemical tanker' at a median 34 m). The object table carries the association WITH quality
fields per detection -- match_status, matched_mmsi, candidate_distance_m, ais_unique_match -- and the trustworthy
subset is match_status == 'unique_spatial_candidate', which is also where these chips' labels came from. Redone here
with that filter, then the same two questions: does length separate the classes the way ships do, and does it
separate known from unknown chips?
"""
import csv
import gzip
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
DS = ROOT / 'dataset244'
OBJ = KS / 'objects/objects_classed.csv.gz'
LOOK = KS / 'mmsi/mmsi_sarais_lookup.csv'
OUT = ROOT / 'features_244'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
want = set((r['product'], r['pol'], r['det']) for r in idx)

dim = {}
with LOOK.open(encoding='utf-8-sig') as fh:
    for row in csv.DictReader(fh):
        m = (row.get('mmsi') or '').strip()
        try:
            L = float(row.get('length') or 0); W = float(row.get('width') or 0)
        except ValueError:
            continue
        if m and L > 0:
            dim[m] = (L, W)
print('尺寸表 %d 个 MMSI' % len(dim), flush=True)

mstat = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        prod = row['object_id'].split('|')[0]
        k = (prod, row.get('polarization'), row.get('detection_id'))
        if k not in want or k in mstat:
            continue
        if (row.get('match_status') or '').strip() != 'unique_spatial_candidate':
            continue
        m = (row.get('matched_mmsi') or '').strip()
        if not m or m == '0':
            continue
        try:
            d = float((row.get('candidate_distance_m') or '[]').strip('[]') or 0)
        except ValueError:
            d = np.nan
        mstat[k] = (m, d)
print('unique 关联命中 %d / %d 芯片键' % (len(mstat), len(want)), flush=True)

L = np.full(len(idx), np.nan); dist = np.full(len(idx), np.nan)
for i, r in enumerate(idx):
    v = mstat.get((r['product'], r['pol'], r['det']))
    if v and v[0] in dim:
        L[i] = dim[v[0]][0]
        dist[i] = v[1]
cov = int(np.isfinite(L).sum())
print('芯片尺寸覆盖 %d / %d (%.0f%%)  |  关联距离中位 %.0f m' % (
    cov, len(idx), 100 * cov / len(idx), np.nanmedian(dist)), flush=True)
np.save(OUT / 'ais_length_244.npy', L)

print('')
print('自检：各类 AIS 登记长度')
print('%-24s %6s %8s %8s %8s' % ('class', 'n', '中位长(m)', 'p25', 'p75'))
for c in sorted(set(cls.tolist()), key=lambda x: -int((cls == x).sum())):
    m = (cls == c) & np.isfinite(L)
    if m.sum() < 10:
        continue
    v = L[m]
    print('%-24s %6d %8.0f %8.0f %8.0f' % (c, int(m.sum()), np.median(v), np.percentile(v, 25), np.percentile(v, 75)))

kn = L[known & np.isfinite(L)]; un = L[(~known) & np.isfinite(L)]
print('')
print('已知 n=%d 中位 %.0f m | 未知 n=%d 中位 %.0f m' % (
    len(kn), np.median(kn) if len(kn) else -1, len(un), np.median(un) if len(un) else -1))
if len(kn) > 20 and len(un) > 20:
    print('★ 长度对"已知 vs 未知"的 AUC: %.3f' % roc_auc_score(np.r_[np.zeros(len(kn)), np.ones(len(un))], np.r_[kn, un]))
    print('  逐未知类中位长度:')
    for c in sorted(set(cls[~known].tolist())):
        m = (cls == c) & np.isfinite(L)
        if m.sum() >= 10:
            print('    %-24s n=%5d 中位 %6.0f m' % (c, int(m.sum()), np.median(L[m])))

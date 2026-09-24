"""e103: per-vessel AIS dimensions from the LOCAL per-product AIS records -- the axis that was data-blocked.

Where the data is (verified, no download needed):
  per-product cached copy : annotation_bundle/assets/products/<product>/ais/NN.txt
  original source          : E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/products_by_primary_port/<port>/
Record layout, 18 comma-separated fields, read off the header line:
  [0] epoch [1] mmsi [2] [3] [4] [5] [6] lon [7] lat [8] [9] [10] name [11] type_code [12] imo [13] callsign
  [14] [15] [16] dimensions (AIS class-A static: to_bow / to_stern / to_port / to_starboard)
  [17] a source tag
Two interpretations of the length are carried through: L_sum = [14]+[15] (the standard reading) and L_bow = [14].

Association: the object table's matched_mmsi restricted to match_status == 'unique_spatial_candidate' -- the same
AIS-matched tier the labels come from -- joined to the chips by (product, pol, det). Only the physical dimensions are
used, never the type code.

Read-outs: per-class length (sanity: tugs short, bulk/container long) and the AUC separating known from unknown
chips.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
DS = ROOT / 'dataset244_q'
BUNDLE = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/products')
OBJ = KS / 'objects/objects_classed.csv.gz'
OUT = ROOT / 'features_244'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
want = set((r['product'], r['pol'], r['det']) for r in idx)
print('chips %d | known %d | unknown %d' % (len(idx), int(known.sum()), int((~known).sum())), flush=True)

# ---- MMSI -> dimensions, from the per-product AIS records ----
dim = {}
nfiles = 0
for prod_dir in BUNDLE.iterdir():
    if not prod_dir.is_dir():
        continue
    adir = prod_dir / 'ais'
    if not adir.is_dir():
        continue
    for f in sorted(adir.glob('*.txt')):
        nfiles += 1
        try:
            with f.open(encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    p = line.rstrip('\n').split(',')
                    if len(p) < 17:
                        continue
                    m = p[1].strip()
                    if not m or m in dim:
                        continue
                    try:
                        a, b, c = int(p[14] or 0), int(p[15] or 0), int(p[16] or 0)
                    except ValueError:
                        continue
                    if a + b > 0:
                        dim[m] = (a + b, a, max(c, 0))
        except OSError:
            continue
    if nfiles % 200 == 0:
        print('  已扫 %d 个 AIS 文件，MMSI %d' % (nfiles, len(dim)), flush=True)
print('AIS 文件 %d 个 | MMSI 尺寸 %d 个' % (nfiles, len(dim)), flush=True)

# ---- association via the AIS-matched tier ----
mstat = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        if (row.get('match_status') or '').strip() != 'unique_spatial_candidate':
            continue
        k = (row['object_id'].split('|')[0], row.get('polarization'), row.get('detection_id'))
        if k not in want or k in mstat:
            continue
        m = (row.get('matched_mmsi') or '').strip()
        if m and m != '0':
            mstat[k] = m
print('unique 关联命中 %d / %d' % (len(mstat), len(want)), flush=True)

L = np.full((len(idx), 2), np.nan, np.float32)
covL, covB = 0, 0
for i, r in enumerate(idx):
    m = mstat.get((r['product'], r['pol'], r['det']))
    if m and m in dim:
        L[i, 0] = dim[m][0]; L[i, 1] = dim[m][1]
        covL += 1
print('尺寸覆盖 %d / %d (%.0f%%)' % (covL, len(idx), 100 * covL / len(idx)), flush=True)
np.save(OUT / 'ais_dim_244q.npy', L)

print('')
print('自检：各类的 AIS 登记长度（L=[14]+[15]，米）')
print('%-24s %6s %8s %8s %8s' % ('class', 'n', '中位', 'p25', 'p75'))
for c in sorted(set(cls.tolist()), key=lambda x: -int((cls == x).sum())):
    m = (cls == c) & np.isfinite(L[:, 0])
    if m.sum() < 10:
        continue
    v = L[m, 0]
    print('%-24s %6d %8.0f %8.0f %8.0f' % (c, int(m.sum()), np.median(v), np.percentile(v, 25), np.percentile(v, 75)))

kn = L[known & np.isfinite(L[:, 0]), 0]
un = L[(~known) & np.isfinite(L[:, 0]), 0]
print('')
print('已知 n=%d 中位 %.0f m | 未知 n=%d 中位 %.0f m' % (len(kn), np.median(kn) if len(kn) else -1,
                                                        len(un), np.median(un) if len(un) else -1))
if len(kn) > 20 and len(un) > 20:
    print('★ 长度对"已知 vs 未知"的 AUC: %.3f   （此前最好 0.549）' % roc_auc_score(
        np.r_[np.zeros(len(kn)), np.ones(len(un))], np.r_[kn, un]))
    print('  逐未知类中位长度:')
    for c in sorted(set(cls[~known].tolist())):
        m = (cls == c) & np.isfinite(L[:, 0])
        if m.sum() >= 10:
            print('    %-24s n=%5d 中位 %6.0f m' % (c, int(m.sum()), np.median(L[m, 0])))

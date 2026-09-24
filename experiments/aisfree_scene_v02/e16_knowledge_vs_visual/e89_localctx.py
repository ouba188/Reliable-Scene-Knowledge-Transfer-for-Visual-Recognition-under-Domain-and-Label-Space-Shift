"""e89: local-scene-context knowledge, built from the object table alone -- then the NAR knowledge half rerun.

Why: OSM facility layers only exist for 3 of the 8 chip ports, capping knowledge coverage at 1.5% (E17u). The
local scene context needs no facility layer and no coordinate reconciliation -- relative distances hold in any
consistent frame, so the object table's own world_x/world_y suffice. It is also the same family the closed-set
work found the signal in (the contrast features, E17i), and it is LABEL-FREE: only positions, counts and sizes of
neighbouring detections are used, never their class.

Features per chip (in metres, Web-Mercator-ish units are fine for relative distances):
  n250, n500, n1000, n2000   neighbour counts at four radii
  nn_dist                    distance to the nearest other detection
  dens_contrast              local density / port density  (the background-contrast form E17i pointed at)
  nb_len_mean, nb_len_std    the neighbours' size statistics (queue/formation structure)

Then: 8-port LOO, visual-only vs visual+context, and the share of UNKNOWN chips pushed into the liquid known
family, stratified by the local context.
"""
import csv
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
OUT = ROOT / 'features_244'
OUT.mkdir(parents=True, exist_ok=True)
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ_IDX = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
RADII = [250.0, 500.0, 1000.0, 2000.0]
MODE = sys.argv[1] if len(sys.argv) > 1 else 'both'

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))
       if r['port'] and (DS / ('%s_%s.npy' % (r['port'].replace(' ', '_'), r['pol']))).is_file()]
print('chips', len(idx), flush=True)
want = set((r['product'], r['pol'], r['det']) for r in idx)

# ---- object table: coordinates + sizes for every detection keyed like the index ----
print('scanning the object table ...', flush=True)
xy, sz = {}, {}
byport_xy = defaultdict(list)
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        prod = row['object_id'].split('|')[0]
        pol = row.get('polarization'); det = row.get('detection_id')
        pj = row.get('port')
        try:
            x, y = float(row['world_x']), float(row['world_y'])
        except (TypeError, ValueError, KeyError):
            continue
        try:
            s = float(row.get('area_px2') or 0) ** 0.5
        except (TypeError, ValueError):
            s = 0.0
        byport_xy[pj].append((x, y, s))
        if (prod, pol, det) in want:
            xy[(prod, pol, det)] = (x, y)
            sz[(prod, pol, det)] = s
print('对象表坐标 %d 条 | 本索引命中 %d/%d' % (sum(len(v) for v in byport_xy.values()), len(xy), len(want)), flush=True)

trees = {}
for pj, pts in byport_xy.items():
    if len(pts) < 50:
        continue
    trees[pj] = cKDTree(np.array([[p[0], p[1]] for p in pts]))
print('港口 KDTree %d 个' % len(trees), flush=True)

K = np.full((len(idx), 8), np.nan)
for i, r in enumerate(idx):
    v = xy.get((r['product'], r['pol'], r['det']))
    if v is None or r['port'] not in trees:
        continue
    t = trees[r['port']]; q = np.array([v[0], v[1]])
    cnts = [len(t.query_ball_point(q, R)) - 1 for R in RADII]
    d, j = t.query(q, k=2)
    nn = d[1] if len(d) > 1 else np.nan
    nport = len(byport_xy[r['port']])
    a = np.pi * RADII[2] ** 2
    dens = cnts[2] / max(1e-9, a)
    dp = nport / max(1e-9, a * 50)
    nb = t.query_ball_point(q, RADII[1])
    lens = np.array([byport_xy[r['port']][k][2] for k in nb], float)
    K[i] = [cnts[0], cnts[1], cnts[2], cnts[3], nn, dens / max(1e-9, dp),
            np.nanmean(lens) if len(lens) else np.nan, np.nanstd(lens) if len(lens) > 1 else np.nan]
print('局部上下文可算 %d / %d' % (int(np.isfinite(K[:, 0]).sum()), len(idx)), flush=True)
np.save(OUT / 'localctx_244.npy', K)

if MODE == 'ctx':
    sys.exit(0)

# ---- the NAR knowledge half, with the local context as the knowledge ----
X = np.load(OUT / 'resnet50_s1b_244.float16.npy').astype(np.float32)
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
Kf = np.log1p(np.nan_to_num(K, nan=0.0))
have = np.isfinite(K[:, 0])
print('已知 %d | 未知 %d | 有上下文 %d' % (int(known.sum()), int((~known).sum()), int(have.sum())), flush=True)

rows = []
for p in sorted(set(ports[known])):
    tr = known & (ports != p) & have
    if tr.sum() < 200:
        continue
    te = (ports == p) & have
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[tr], y[tr])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[X[tr], Kf[tr]], y[tr])
    ti = np.where(te)[0]
    pv = rv.predict(X[ti]); pk = rk.predict(np.c_[X[ti], Kf[ti]])
    for pos, i in enumerate(ti):
        rows.append({'i': int(i), 'cls': cls[i], 'known': bool(known[i]),
                     'pred_v': int(pv[pos]), 'pred_k': int(pk[pos]),
                     'n1k': K[i, 2], 'dens': K[i, 5]})
print('评估 %d' % len(rows))
kn = [r for r in rows if r['known']]; un = [r for r in rows if not r['known']]
print('')
print('已知类准确率（自检）：视觉 %.3f  视觉+上下文 %.3f' % (
    np.mean([r['pred_v'] == y[r['i']] for r in kn]), np.mean([r['pred_k'] == y[r['i']] for r in kn])))
print('')
print('未知片被预测为液货族(crude/pchem)：视觉 %.3f  视觉+上下文 %.3f' % (
    np.mean([r['pred_v'] in LIQ_IDX for r in un]), np.mean([r['pred_k'] in LIQ_IDX for r in un])))
q = np.quantile([r['n1k'] for r in un], [1 / 3, 2 / 3])
for nm, g in (('稀疏(下1/3)', [r for r in un if r['n1k'] <= q[0]]),
              ('中等', [r for r in un if q[0] < r['n1k'] <= q[1]]),
              ('密集(上1/3)', [r for r in un if r['n1k'] > q[1]])):
    if not g:
        continue
    print('%-12s n=%5d  视觉 %.3f → 视觉+上下文 %.3f  (Δ %+.3f)' % (
        nm, len(g), np.mean([r['pred_v'] in LIQ_IDX for r in g]),
        np.mean([r['pred_k'] in LIQ_IDX for r in g]),
        np.mean([r['pred_k'] in LIQ_IDX for r in g]) - np.mean([r['pred_v'] in LIQ_IDX for r in g])))
print('')
for c in ('lpg_lng_tanker', 'dredger', 'crude_oil_tanker', 'bulk_carrier'):
    g = [r for r in un if r['cls'] == c]
    if not g:
        continue
    print('%-20s n=%5d  视觉 %.3f → 视觉+上下文 %.3f' % (
        c, len(g), np.mean([r['pred_v'] in LIQ_IDX for r in g]), np.mean([r['pred_k'] in LIQ_IDX for r in g])))
np.save(OUT / 'e89_res.npy', np.array(rows, dtype=object), allow_pickle=True)

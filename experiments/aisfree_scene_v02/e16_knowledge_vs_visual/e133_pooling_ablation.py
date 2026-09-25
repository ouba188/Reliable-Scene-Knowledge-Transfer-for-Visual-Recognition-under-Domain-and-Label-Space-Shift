"""e133: is cross-acquisition COVERAGE the value? -- the same field recipe, two poolings.

E18t/E18v: the AIS knowledge block's value comes from coverage (a port-wide field), the per-scene SAR block carries a tenth of
it, and a SAR-only port field is feasible at ~2 km cells. The chip<->object join needed to attach such a field to the old
pool's visual features is blocked (the manifest's source_object_index is empty for legacy rows and (product, polarity, det) is
not unique in the object table), so this measures the mechanism directly on the object table itself, which needs no join:

  same 11 field quantities, computed twice per object --
    per_scene : pooled over the object's OWN product only   (what a single acquisition can see)
    pooled    : pooled over the whole port, all acquisitions (what SAR can accumulate over time)
  target: the object's AIS-derived fine class, 8 known classes; 24-port leave-one-port-out; ridge.

If coverage is the value, the pooled arm must beat the per-scene arm on the same recipe, same dims, same folds.
Pre-registered: BA(pooled) - BA(per_scene) > 0 with paired p < 0.05 over the 24 ports.
"""
import array
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import os

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeClassifier

OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
KNOWN8 = ['bulk_carrier', 'container_ship', 'crude_oil_tanker', 'fishing_vessel',
          'general_cargo', 'offshore_supply', 'product_chemical_tanker', 'tug_towing']
POOL_POL = 'VH'
LABEL = os.environ.get('LABEL_COL', 'prelabel_class')   # fine_class covers only 11,450 rows in the 8 known classes; prelabel_class covers 123,790
R5, R10 = 5000.0, 10000.0

rows = []
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='replace') as f:
    rd = csv.DictReader(f)
    for r in rd:
        c = (r.get(LABEL) or '').strip()
        if c not in KNOWN8:
            continue
        try:
            x = float(r['world_x']); y = float(r['world_y'])
        except (KeyError, ValueError):
            continue
        h = float(r['obb_long_deg']) if (r.get('obb_long_deg') or '').strip() else float('nan')
        q = float(r['d_quay_m']) if (r.get('d_quay_m') or '').strip() else float('nan')
        rows.append((r.get('port') or '?', (r.get('polarization') or '').upper(), r.get('product_id') or '',
                     x, y, h, q, KNOWN8.index(c)))
print('标签列 %s' % LABEL); print('已知 8 类对象 %d ｜ 港 %d ｜ 产品 %d' % (len(rows), len({r[0] for r in rows}), len({r[2] for r in rows})), flush=True)

port = np.array([r[0] for r in rows]); pol = np.array([r[1] for r in rows])
prod = np.array([r[2] for r in rows]); y = np.array([r[7] for r in rows])
X = np.array([[r[3], r[4]] for r in rows], np.float32)
HD = np.array([r[5] for r in rows], np.float32); QD = np.array([r[6] for r in rows], np.float32)

# pooled storage: per port (all acquisitions, one polarisation) and per product (single acquisition)
byport, byprod = defaultdict(lambda: [[], []]), defaultdict(lambda: [[], []])
for i in range(len(rows)):
    if pol[i] != POOL_POL:
        continue
    byport[port[i]][0].append(X[i]); byport[port[i]][1].append(HD[i])
    byprod[prod[i]][0].append(X[i]); byprod[prod[i]][1].append(HD[i])


def make(store):
    T, H, C = {}, {}, {}
    for k, (pts, hh) in store.items():
        A = np.array(pts, np.float32)
        cen = np.median(A, 0)
        A = A - cen
        T[k] = cKDTree(A); H[k] = np.array(hh, np.float32); C[k] = cen
    return T, H, C


TP, HP, CP = make(byport)
TS, HS, CS = make(byprod)
print('KD 树：港 %d ｜ 产品 %d' % (len(TP), len(TS)), flush=True)


def field(T, H, C, key, pts, qp):
    t = T[key]
    n2 = np.array([len(t.query_ball_point(p, 2000.0)) for p in pts], np.float32)
    n5 = t.query_ball_point(pts, R5, return_length=True).astype(np.float32)
    n10 = t.query_ball_point(pts, R10, return_length=True).astype(np.float32)
    nb = t.query_ball_point(pts, R5)
    hh = H[key]
    s2 = np.zeros(len(pts), np.float32); co2 = np.zeros(len(pts), np.float32)
    for i, ix in enumerate(nb):
        if len(ix):
            th = np.radians(np.nan_to_num(hh[ix], nan=0.0)) * 2.0
            s2[i] = float(np.sin(th).mean()); co2[i] = float(np.cos(th).mean())
    conc = np.hypot(s2, co2)
    c2, c5, c10 = np.log1p(n2), np.log1p(n5), np.log1p(n10)
    qprox = np.exp(-np.nan_to_num(qp, nan=R10) / 2000.0)
    return np.c_[c2 - c5, c5 - c10, c10, n2 / np.maximum(n10, 1.0), s2, co2, conc,
                 c2 * conc, qprox, c2 * qprox, conc * qprox].astype(np.float32)


idx = np.arange(len(rows))
Fp = np.zeros((len(rows), 11), np.float32)
Fs = np.zeros((len(rows), 11), np.float32)
for p in sorted(set(port.tolist())):
    sel = idx[port == p]
    if p not in TP:
        continue
    Fp[sel] = field(TP, HP, CP, p, X[sel] - CP[p], QD[sel])
    for pr in sorted(set(prod[sel].tolist())):
        s2i = sel[prod[sel] == pr]
        if pr in TS:
            Fs[s2i] = field(TS, HS, CS, pr, X[s2i] - CS[pr], QD[s2i])
    print('  %-18s n=%6d' % (p, len(sel)), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs))


ARMS = {'场(全港池化)': Fp, '场(本景池化)': Fs, '两者': np.c_[Fp, Fs]}
res = {a: [] for a in ARMS}
for p in sorted(set(port.tolist())):
    tr = np.where(port != p)[0]; te = np.where(port == p)[0]
    if len(tr) < 2000 or len(te) < 50:
        continue
    for a, Q in ARMS.items():
        mu = Q[tr].mean(0, keepdims=True); sd = Q[tr].std(0, keepdims=True) + 1e-6
        m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Q[tr] - mu) / sd, y[tr])
        res[a].append(ba(y[te], m.predict((Q[te] - mu) / sd)))

print('')
print('%-16s %9s %9s' % ('arm', 'BA', '港数'))
for a in ARMS:
    print('%-16s %9.4f %9d' % (a, np.mean(res[a]), len(res[a])))
p_, s_ = np.array(res['场(全港池化)']), np.array(res['场(本景池化)'])
print('')
print('配对 全港 − 本景: %+.4f (%.2fpp)  p=%.4f ｜ 正港 %d/%d'
      % ((p_ - s_).mean(), (p_ - s_).mean() * 100,
         stats.wilcoxon(p_, s_).pvalue if np.any(p_ != s_) else float('nan'),
         int((p_ > s_).sum()), len(p_)))
print('预注册判据（全港 > 本景 且 p<0.05）: %s'
      % ('成立 ✓✓ —— 覆盖确实是价值来源 ✓' if ((p_ - s_).mean() > 0 and stats.wilcoxon(p_, s_).pvalue < 0.05) else '未成立 ✗'))

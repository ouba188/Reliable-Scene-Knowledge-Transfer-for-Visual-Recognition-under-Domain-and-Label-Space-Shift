"""e118: is one latent axis -- class distinctiveness -- behind BOTH phenomena?

E17i showed facility sharing predicts how much knowledge HURTS a class (AUC 0.72 per chip, rho=-0.762 across 8 classes).
E18i showed novelty detectability is a class property (across-class SD 2.3x the bootstrap noise) and that the classes which
fail detection are the high-sharing ones. Both entries used a proxy (facility-sharing entropy) built on the 8-class pool.

This measures the axis DIRECTLY, in visual space, on the 24-port pool, and correlates it with both phenomena:
  distinctiveness_c = mean over (port, instance) of [ Mahalanobis distance to the nearest OTHER class centroid
                                                   - Mahalanobis distance to its own class centroid ]
  Y1 = per-class novelty AUC (e117, 9 unknown classes)          expect +
  Y2 = per-class dVK (E17i, 8 known classes)                    expect +
Same sign on both = one axis governs both phenomena. Pre-registered: rho > 0 on both, permutation p < 0.05 on Y1.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
X = np.load(ROOT / 'features_all' / 'resnet50_c64.float16.npy').astype(np.float32)
rows = list(csv.DictReader((ROOT / 'dataset_all' / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
# E17i per-class dVK (knowledge arm minus visual arm, percentage points, old 38,091 pool -- same classes, noted as a caveat)
DVK = {'bulk_carrier': 15.6, 'general_cargo': 2.3, 'tug_towing': 6.8, 'offshore_supply': -2.2,
       'fishing_vessel': -1.0, 'product_chemical_tanker': -5.1, 'crude_oil_tanker': -13.9, 'container_ship': -3.4}
NOV = json.loads((ROOT / 'e117_perclass.json').read_text(encoding='utf-8'))

CLASSES = sorted(set(cls.tolist()))
mar = {}
for p in sorted(set(ports.tolist())):
    te = np.where(ports == p)[0]
    if len(te) < 500:
        continue
    Z = X[te]
    mu = Z.mean(0, keepdims=True)
    Zc = Z - mu
    C = Zc.T @ Zc / len(Z)
    C = 0.95 * C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])   # ponytail: same shrinkage as e117; small ports are rank-deficient
    P = np.linalg.inv(C.astype(np.float64))
    cen = {}
    for c in CLASSES:
        m = cls[te] == c
        if m.sum() >= 20:
            cen[c] = Z[m].mean(0)
    for c, mc in cen.items():
        own = ((Z[cls[te] == c] - mc) @ P * (Z[cls[te] == c] - mc)).sum(1)
        other = np.full_like(own, np.inf)
        for c2, mc2 in cen.items():
            if c2 == c:
                continue
            d2 = ((Z[cls[te] == c] - mc2) @ P * (Z[cls[te] == c] - mc2)).sum(1)
            other = np.minimum(other, d2)
        mar.setdefault(c, []).append(float(np.mean(np.sqrt(np.maximum(own, 0)) - np.sqrt(np.maximum(other, 0)))))
    print('%-16s distinctiveness: %s' % (p, ' '.join('%s %+.1f' % (c[:6], mar[c][-1]) for c in cen)), flush=True)

dist = {c: float(np.mean(v)) for c, v in mar.items()}
print('')
print('%-26s %12s %10s %10s' % ('class', 'distinct.', 'novelty', 'dVK(pp)'))
xs1, ys1, xs2, ys2 = [], [], [], []
for c in CLASSES:
    if c not in dist:
        continue
    nv = NOV.get(c); dv = DVK.get(c)
    print('%-26s %12.3f %10s %10s' % (c, dist[c], '%.3f' % nv if nv else '-', '%.1f' % dv if dv else '-'))
    if nv:
        xs1.append(dist[c]); ys1.append(nv)
    if dv is not None:
        xs2.append(dist[c]); ys2.append(dv)
print('')
for nm, xs, ys in (('distinctiveness vs novelty AUC (9 unknown)', xs1, ys1),
                   ('distinctiveness vs dVK (8 known)', xs2, ys2)):
    sp = stats.spearmanr(xs, ys)
    pm = float((np.abs([stats.spearmanr(xs, np.random.permutation(ys)).statistic for _ in range(5000)]) >= abs(sp.statistic)).mean())
    print('%-46s rho=%+.3f p=%.3f 置换 p=%.4f n=%d' % (nm, sp.statistic, sp.pvalue, pm, len(xs)))
print('')
same = np.sign(stats.spearmanr(xs1, ys1).statistic) == np.sign(stats.spearmanr(xs2, ys2).statistic)
print('预注册判据（两条 rho 同号为正）：%s' % ('统一轴成立 ✓✓' if (same and stats.spearmanr(xs1, ys1).statistic > 0) else '未成立 ✗'))

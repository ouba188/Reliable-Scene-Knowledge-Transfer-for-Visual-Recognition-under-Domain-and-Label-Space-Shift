"""e73: which knowledge BLOCK carries the e69 signal? (naming the mechanism honestly)

e69's entropy measure works (AUC ~0.75) but the liquid/dry chain interpretation failed (e72: at chip level
crude sits FURTHER from the liquid-chain dims than bulk does). So decompose: recompute the same sharing
measure using only one block at a time, and see which block reproduces the discrimination.

Blocks (relation_features.json):
  0-13    OSM facility proximity (oil tanks, silos, cranes, quay, pipelines, shipyard, ...)
  14-41   the same facilities as background-contrast / near-vs-route variants
  42-59   operational-chain composites (liquid/dry/general/industrial/nav)
  60-70   SAR detection-side spatial field
  71-81   AIS-side field (leakage, never legal)
Outcome: harm vs rescue from the frozen e61 predictions.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1

BLOCKS = {
    'facility_prox(0-13)': list(range(0, 14)),
    'facility_contrast(14-41)': list(range(14, 42)),
    'chains(42-59)': list(range(42, 60)),
    'detection(60-70)': list(range(60, 71)),
    'LEGAL_all(0-70)': list(range(0, 71)),
    'ais_field(71-81)': list(range(71, 82)),
}

Kp = np.zeros((len(y), 82))
for pj in sorted(set(ports.tolist())):
    m = ports == pj
    Kp[np.ix_(m, range(82))] = K[np.ix_(m, range(82))].argsort(0).argsort(0) / max(1, int(m.sum()) - 1)

d61 = pickle.load(open('e61_preds.pkl', 'rb'))     # our own artifact
resc = np.zeros(len(y), bool); harm = np.zeros(len(y), bool)
for rec in d61:
    sel = np.where(ports == rec['port'])[0]
    v, vk, yy = rec['pred']['ridge_V'], rec['pred']['ridge_VK'], rec['y']
    resc[sel] = (v != yy) & (vk == yy)
    harm[sel] = (v == yy) & (vk != yy)
sel = harm | resc
print('非中性 chip %d（救回 %d / 误伤 %d）' % (sel.sum(), resc.sum(), harm.sum()))
print()
print('%-26s %8s %10s %10s' % ('block', '可用维', 'AUC(harm)', '含义'))
res = []
for nm, ds in BLOCKS.items():
    # per-dim class entropy over the chips in that dim's top quartile, from the SOURCE-agnostic pooled view
    Hd = {}
    for d in ds:
        nz = K[K[:, d] != 0, d]
        if nz.size < 200:
            continue
        m = ((K[:, d] >= np.quantile(nz, 0.75)) * (K[:, d] != 0).astype(int)).astype(bool)
        if m.sum() < 200:
            continue
        p = np.bincount(y[m], minlength=C).astype(float); p /= p.sum(); p = p[p > 0]
        Hd[d] = float(-(p * np.log(p)).sum() / np.log(C))
    sh = np.full(len(y), np.nan)
    for i in range(len(y)):
        vs = [Hd[d] for d in ds if d in Hd and K[i, d] != 0]
        if vs:
            sh[i] = float(np.mean(vs)) if len(vs) else np.nan
    ok = np.isfinite(sh) & sel
    if ok.sum() < 200:
        print('%-26s %8d %10s' % (nm, len(Hd), 'n/a'))
        continue
    a = roc_auc_score(harm[ok], sh[ok])
    res.append((nm, len(Hd), a, ok.sum()))
    print('%-26s %8d %10.3f   n=%d' % (nm, len(Hd), a, ok.sum()))
print()
print('（AUC>0.5 ⇒ 该块的"共享度"越高越容易被伤害；LEGAL_all 应复现 e69 的 ~0.61）')

"""e177: known-vs-unknown head-to-head -- the classifier's margin (E21a's by-product) against Mahalanobis novelty (e117).

E21a showed the margin ranking concentrates known classes (0.92 known among the top 10% against 0.84 random). The established
novelty score in this project is the class-conditional Mahalanobis distance (e117: mean per-class AUC 0.555, across-class SD 2.3x
its bootstrap noise). Both are computable in one pass and neither needs a target label, so the honest question is which one
separates known from unknown better -- per unknown class and pooled, with a paired test.

Scores: margin = top1 - top2 of the 17-class source-trained ridge (higher = more confident = more likely known);
        novelty = min Mahalanobis distance to the source class centroids for the KNOWN classes (higher = more novel).
AUC per (port, unknown class) against that port's known instances; pooled per class across ports.
Pre-registered: whichever score wins must beat the other with paired p < 0.05 over the (port, class) cells.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
X = np.load(BASE / 'features_all' / 'pca512_all.npy').astype(np.float32)
rows = list(csv.DictReader((BASE / 'dataset_all' / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
names = sorted(set(cls.tolist()))
y = np.array([names.index(c) for c in cls])
C = len(names)
is_known = np.isin(cls, KNOWN8)
UNK = sorted(set(cls[~is_known].tolist()))
print('新池 %d 片 ｜ 已知 %d / 未知 %d ｜ 未知类 %d' % (len(rows), int(is_known.sum()), int((~is_known).sum()), len(UNK)), flush=True)

cell = defaultdict(lambda: {'m': [], 'n': []})
for p in sorted(set(ports.tolist())):
    src = np.where(is_known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    mu = X[src].mean(0, keepdims=True); sd = X[src].std(0, keepdims=True) + 1e-6
    Z = (X[src] - mu) / sd; Q = (X[te] - mu) / sd
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[src])
    L = m.decision_function(Q)
    if L.shape[1] != C:
        Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    # class-conditional Gaussians on the source's KNOWN classes only
    Zk = (X[src] - mu) / sd
    cen = np.stack([Zk[y[src] == j].mean(0) for j in range(C) if (y[src] == j).any()])
    lab = [j for j in range(C) if (y[src] == j).any()]
    Cc = np.cov(Zk.T) if Zk.shape[0] > Zk.shape[1] else np.cov(Zk.T) + 1e-3 * np.eye(Zk.shape[1])
    P = np.linalg.inv(Cc + 1e-3 * np.eye(Cc.shape[0]))
    d = np.stack([np.einsum('ij,jk,ik->i', Q - cen[i], P, Q - cen[i]) for i in range(len(lab))])
    novelty = d.min(0)
    kn = is_known[te]
    for c in UNK:
        sel = cls[te] == c
        if sel.sum() < 5 or kn.sum() < 5:
            continue
        cell[(p, c)]['m'].append(roc_auc_score(sel.astype(int), margin))
        cell[(p, c)]['n'].append(roc_auc_score(sel.astype(int), novelty))
    print('%-16s 已知 %5d 未知 %5d' % (p, int(kn.sum()), int((~kn).sum())), flush=True)

am = {c: [] for c in UNK}; an = {c: [] for c in UNK}
pairs = []
for (p, c), v in cell.items():
    if v['m'] and v['n']:
        am[c].append(v['m'][0]); an[c].append(v['n'][0]); pairs.append((v['m'][0], v['n'][0]))
print('')
print('%-26s %5s %10s %10s' % ('unknown class', '港数', 'margin AUC', 'novelty AUC'))
for c in UNK:
    if am[c]:
        print('%-26s %5d %10.3f %10.3f' % (c, len(am[c]), np.mean(am[c]), np.mean(an[c])))
M = np.array([np.mean(am[c]) for c in UNK if am[c]])
N = np.array([np.mean(an[c]) for c in UNK if an[c]])
print('')
print('跨类均值：margin %.3f ｜ novelty %.3f ｜ 差 %+.3f' % (M.mean(), N.mean(), (M - N).mean()))
pp = np.array(pairs)
if len(pp) >= 8:
    pv = stats.wilcoxon(pp[:, 0], pp[:, 1]).pvalue
    print('逐 (港,类) 配对 n=%d：Δ %+.4f ｜ p=%.4f ⇒ %s'
          % (len(pp), (pp[:, 0] - pp[:, 1]).mean(), pv,
             'margin 更优 ✓✓' if (pp[:, 0].mean() > pp[:, 1].mean() and pv < 0.05) else
             ('novelty 更优 ✓✓' if (pp[:, 1].mean() > pp[:, 0].mean() and pv < 0.05) else '无显著差异 ⚠')))
else:
    print('配对样本不足（%d）⚠' % len(pp))

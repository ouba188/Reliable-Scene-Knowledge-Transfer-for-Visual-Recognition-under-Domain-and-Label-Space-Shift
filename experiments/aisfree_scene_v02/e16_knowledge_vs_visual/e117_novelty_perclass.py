"""e117: per-class novelty detectability (open-set / label-space shift) on the 24-port pool.

The pool carries 8 known classes AND 9 further AIS-labelled classes that the closed-set protocol never uses. Open-set
work so far (e90/e91) only used them as a single lumped "unknown" for a novelty head; the standard strong baselines for
detecting label-space shift -- class-conditional Mahalanobis distance and energy score -- were never run at this scale.
Fitted on source ports only; the target port contributes no fitting data at all.

Per (target port, unknown class) cell: AUC of known-vs-unknown. Pre-registered criterion: the spread of AUC across
unknown classes must exceed the shuffle null, i.e. SOME classes are separable and others are not. If every class sits
at ~0.50 the observation regime admits no novelty signal at all and the module space stays closed.
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy import stats
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / os.environ.get('DS_DIR', 'dataset_all')
FEAT = ROOT / os.environ.get('FEAT_DIR', 'features_all') / 'resnet50_c64.float16.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
SHRINK = 0.05           # ponytail: fixed ridge shrinkage; Ledoit-Wolf only if the rank of the covariance bites
rng = np.random.default_rng(0)

rows = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
is_known = np.isin(cls, KNOWN8)
y = np.full(len(rows), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
X = np.load(FEAT).astype(np.float32)
UNK = sorted(set(cls[~is_known]))
print('chips %d | 已知 %d | 未知 %d | 未知类 %d' % (len(rows), int(is_known.sum()), int((~is_known).sum()), len(UNK)), flush=True)


def auc(pos, neg):
    if len(pos) < 3 or len(neg) < 3:
        return float('nan')
    a = np.concatenate([pos, neg])
    r = stats.rankdata(a)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


grid = {}          # (port, class) -> (auc_maha, auc_energy, n)
for p in sorted(set(ports.tolist())):
    tr = np.where(is_known & (ports != p))[0]
    if len(tr) < 1000:
        continue
    Z = X[tr]
    mu_all = Z.mean(0, keepdims=True)
    means = np.stack([Z[y[tr] == j].mean(0) for j in range(8)])
    Zc = Z - means[y[tr]]
    C = Zc.T @ Zc / len(Z)
    C = (1 - SHRINK) * C + SHRINK * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C.astype(np.float64)).astype(np.float32)
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[tr])
    te = np.where(ports == p)[0]
    Q = X[te]
    d = np.stack([np.einsum('ij,jk,ik->i', Q - means[j], P, Q - means[j]) for j in range(8)])
    maha = d.min(0)
    en = logsumexp(clf.decision_function(Q), 1)
    kn = is_known[te]
    for c in UNK:
        m = cls[te] == c
        if m.sum() < 3:
            continue
        grid[(p, c)] = (auc(maha[m], maha[kn]), auc(en[m], en[kn]), int(m.sum()))
    print('%-16s 已知 %5d 未知 %5d | 该港逐类 maha AUC %s' % (
        p, int(kn.sum()), int((~kn).sum()),
        ' '.join('%s %.2f' % (c[:6], grid[(p, c)][0]) for c in UNK if (p, c) in grid)), flush=True)

print('')
print('=== 逐未知类（跨港池化）===')
print('%-26s %5s %9s %9s %9s %9s' % ('unknown class', 'n', 'maha AUC', '95%CI', 'energy', '港数'))
per_class = {}
for c in UNK:
    a = np.array([v[0] for k, v in grid.items() if k[1] == c and np.isfinite(v[0])])
    e = np.array([v[1] for k, v in grid.items() if k[1] == c and np.isfinite(v[1])])
    n = sum(v[2] for k, v in grid.items() if k[1] == c)
    if not len(a):
        continue
    lo, hi = np.percentile([np.mean(rng.choice(a, len(a))) for _ in range(500)], [2.5, 97.5])
    per_class[c] = (a.mean(), n, len(a))
    print('%-26s %5d %9.3f %9s %9.3f %9d' % (c, n, a.mean(), '[%.2f,%.2f]' % (lo, hi), e.mean(), len(a)))

print('')
m = np.array([v[0] for v in per_class.values()])
names = list(per_class)
print('跨类均值 %.3f ｜ 最好 %s %.3f ｜ 最差 %s %.3f ｜ 极差 %.3f'
      % (m.mean(), names[int(m.argmax())], m.max(), names[int(m.argmin())], m.min(), m.max() - m.min()))
# null: how much across-class spread would sampling noise alone produce? bootstrap SD per class, averaged.
noise = []
for c in names:
    a = np.array([v[0] for k, v in grid.items() if k[1] == c and np.isfinite(v[0])])
    if len(a) > 2:
        noise.append(np.std([np.mean(rng.choice(a, len(a))) for _ in range(300)]))
noise = float(np.mean(noise)) if noise else float('nan')
print('逐类 bootstrap 噪声 SD %.4f ｜ 跨类实测 SD %.4f ⇒ %s'
      % (noise, m.std(), '跨类差异显著 ✓' if m.std() > 2 * noise else '与噪声不可分 ✗'))
print('')
print('已知 vs 未知 总体（跨港池化）：maha %.3f ｜ energy %.3f'
      % (np.nanmean([v[0] for v in grid.values()]), np.nanmean([v[1] for v in grid.values()])))

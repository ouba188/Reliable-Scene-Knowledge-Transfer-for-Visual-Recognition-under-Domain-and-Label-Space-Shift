"""e178: do the two scores COMBINE? A Pareto question, not a single-metric one.

e176 showed the margin is the best budget-spender for selective accuracy; e177 showed Mahalanobis novelty is the only score that
separates known from unknown (margin is below chance per instance). The honest question is therefore not "which single score
wins" but whether a joint rule DOMINATES both on the two objectives at once, at the same coverage:

    objective 1: BA on the selected subset          (the margin's strength)
    objective 2: known-class rate of the subset     (the novelty's strength)

Rules, all at a 10% budget, on the new pool:
    margin          top 10% by the classifier margin
    novelty         top 10% by LOW novelty (most known-like)
    sum             top 10% by z(margin) + z(-novelty)
    and             intersection of both top 10%s, filled by z(margin)+z(-novelty) if short
    random          floor
Pre-registered: some joint rule must dominate BOTH single-score rules (>= on both objectives, > on one) with paired p < 0.05 on
the dominant objective. If none does, the scores are separately best and must not be presented as combinable.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier

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
K = 0.10
NREP = 30
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-9)


res = {r: {'ba': [], 'kn': []} for r in ('margin', 'novelty', 'sum', 'and', 'random')}
for p in sorted(set(ports.tolist())):
    src = np.where(is_known & (ports != p))[0]; te = np.where(ports == p)[0]
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
    lab = [j for j in range(C) if (y[src] == j).any()]
    cen = np.stack([Z[y[src] == j].mean(0) for j in lab])
    P = np.linalg.inv(np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1]))
    d = np.stack([np.einsum('ij,jk,ik->i', Q - cen[i], P, Q - cen[i]) for i in range(len(lab))])
    novelty = d.min(0)                       # higher = more novel
    pred = L.argmax(1); yy = y[te]; kn = is_known[te]
    n = max(1, int(K * len(te)))
    zm, zn = zs(margin), zs(-novelty)
    top_m = set(np.argsort(-zm)[:n].tolist())
    top_n = set(np.argsort(-zn)[:n].tolist())
    cand = sorted(top_m & top_n, key=lambda i: -(zm[i] + zn[i]))
    if len(cand) < n:
        rest = [i for i in np.argsort(-(zm + zn)).tolist() if i not in cand]
        cand = cand + rest[: n - len(cand)]
    sel = {'margin': np.array(sorted(top_m)), 'novelty': np.array(sorted(top_n)),
           'sum': np.argsort(-(zm + zn))[:n], 'and': np.array(cand),
           'random': rng.permutation(len(te))[:n]}
    for r, idx in sel.items():
        res[r]['ba'].append(ba(yy[idx], pred[idx]))
        res[r]['kn'].append(float(kn[idx].mean()))
    print('%-16s n=%5d' % (p, len(te)), flush=True)

print('')
print('%-9s %10s %12s' % ('rule', 'BA@10%', 'known rate'))
for r in res:
    print('%-9s %10.4f %12.3f' % (r, np.mean(res[r]['ba']), np.mean(res[r]['kn'])))
print('')
for r in ('sum', 'and'):
    for base in ('margin', 'novelty'):
        dba = np.array(res[r]['ba']) - np.array(res[base]['ba'])
        dkn = np.array(res[r]['kn']) - np.array(res[base]['kn'])
        p1 = stats.wilcoxon(res[r]['ba'], res[base]['ba']).pvalue
        p2 = stats.wilcoxon(res[r]['kn'], res[base]['kn']).pvalue
        dom = (np.mean(dba) >= -1e-9 and np.mean(dkn) >= -1e-9) and (np.mean(dba) > 0 or np.mean(dkn) > 0)
        print('%s vs %-8s: ΔBA %+.4f (p=%.4f) ｜ Δknown %+.3f (p=%.4f) ⇒ 支配 %s'
              % (r, base, np.mean(dba), p1, np.mean(dkn), p2, '✓' if dom else '✗'))
print('')
print('预注册判据（某联合规则同时不差于两个单分数且至少一项更优）: 见上表 ✓/✗')

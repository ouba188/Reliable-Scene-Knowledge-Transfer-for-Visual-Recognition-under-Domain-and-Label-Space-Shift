"""e17: pin down the +2.28pp — multi-seed controls, per-class rescue/harm, group ablation.

Clean protocol: alpha chosen per outer fold from source ports only; target port never used.
Legal dims only (0-70; AIS 71-81 excluded as leaky).
"""
import csv, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
feats = json.load((ROOT / 'knowledge' / 'relation_features.json').open(encoding='utf-8'))
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
CLS = [r['class_name'] for r in rows]
name_of = {}
for r in rows:
    name_of[int(r['class_id'])] = r['class_name']
C = int(y.max()) + 1
ALPHAS = [0.1, 0.3, 1.0]
GROUPS = {'V': [], 'facility0-42': list(range(0, 42)), 'chain42-60': list(range(42, 60)),
          'detection60-71': list(range(60, 71)), 'legal0-70': list(range(0, 71)),
          'legal_minus_chain': list(range(0, 42)) + list(range(60, 71)),
          'legal_minus_detection': list(range(0, 60))}


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te, dims, Kd):
    if not dims:
        return X[tr], X[te]
    A, B = Kd[tr][:, dims], Kd[te][:, dims]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def outer(dims, Kd, collect=False):
    out, preds = [], {}
    for p in sorted(set(ports.tolist())):
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
            continue
        inner = sorted(set(ports[tr].tolist()))[:3]
        best_a, best = 1.0, -1.0
        for a in ALPHAS:
            sc = []
            for q in inner:
                itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
                if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                    continue
                f1, f2 = prep(itr, ite, dims, Kd)
                sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[itr]).predict(f2)))
            if sc and float(np.mean(sc)) > best:
                best, best_a = float(np.mean(sc)), a
        ftr, fte = prep(tr, te, dims, Kd)
        pr = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(ftr, y[tr]).predict(fte)
        out.append(ba(y[te], pr))
        if collect:
            preds[p] = (te, pr)
    return (float(np.mean(out)), preds) if collect else float(np.mean(out))


print('=== 单臂 ===')
res = {}
for a, dims in GROUPS.items():
    res[a] = outer(dims, K)
    print('  %-24s %8.4f  Δ %+.4f' % (a, res[a], res[a] - res['V'] if 'V' in res else float('nan')))

print()
print('=== 多种子行打乱对照 (legal0-70) ===')
base = res['legal0-70']
for s in [1, 2, 3]:
    rng = np.random.default_rng(s)
    m = outer(GROUPS['legal0-70'], K[rng.permutation(len(ports))])
    print('  seed %d: %8.4f  Δ vs legal %+.4f  Δ vs V %+.4f' % (s, m, m - base, m - res['V']))

print()
print('=== 逐类 (V vs legal0-70) ===')
_, pv = outer([], K, collect=True)
_, pk = outer(GROUPS['legal0-70'], K, collect=True)
print('  %-26s %8s %8s %9s' % ('class', 'V', 'V+K', 'Δ'))
for c in range(C):
    a = ba(y[np.concatenate([pv[p][0] for p in pv])], np.concatenate([pv[p][1] for p in pv])) if False else None
# 简单逐类：合并所有折
yv = np.concatenate([y[pv[p][0]] for p in pv]); pvv = np.concatenate([pv[p][1] for p in pv])
yk = np.concatenate([y[pk[p][0]] for p in pk]); pkk = np.concatenate([pk[p][1] for p in pk])
for c in range(C):
    m = yv == c
    if m.sum() == 0:
        continue
    rv = float((pvv[m] == c).mean()); rk = float((pkk[m] == c).mean())
    print('  %-26s %8.3f %8.3f %+9.3f   (n=%d)' % (name_of.get(c, c), rv, rk, rk - rv, int(m.sum())))

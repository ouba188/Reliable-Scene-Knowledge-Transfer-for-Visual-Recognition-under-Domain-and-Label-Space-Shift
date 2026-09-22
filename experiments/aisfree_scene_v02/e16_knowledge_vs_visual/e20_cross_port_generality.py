"""e20: is the knowledge cross-port universal? Per-port delta of V -> V+K (clean e17 protocol, no rescaling).

Collects, for each held-out port: BA(V), BA(V+K), and per-class recall changes.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
name_of = {int(r['class_id']): r['class_name'] for r in rows}
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.1, 0.3, 1.0]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te, dims):
    if not dims:
        return X[tr], X[te]
    A, B = K[tr][:, dims], K[te][:, dims]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def pick_alpha(tr, dims):
    inner = sorted(set(ports[tr].tolist()))[:3]
    best_a, best = 1.0, -1.0
    for a in ALPHAS:
        sc = []
        for q in inner:
            itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
            if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                continue
            f1, f2 = prep(itr, ite, dims)
            sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[itr]).predict(f2)))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


print('%-18s %7s %7s %8s   %s' % ('port', 'V', 'V+K', 'Δ', 'per-class Δ (top3)'))
print('-' * 78)
ds = []
percls = {c: [] for c in range(C)}
for p in sorted(set(ports.tolist())):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    a = pick_alpha(tr, LEGAL)
    f1, f2 = prep(tr, te, [])
    g1, g2 = prep(tr, te, LEGAL)
    pv = RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[tr]).predict(f2)
    pk = RidgeClassifier(alpha=a, class_weight='balanced').fit(g1, y[tr]).predict(g2)
    bv, bk = ba(y[te], pv), ba(y[te], pk)
    ds.append(bk - bv)
    ch = []
    for c in range(C):
        m = y[te] == c
        if m.sum() >= 5:
            d = float((pk[m] == c).mean()) - float((pv[m] == c).mean())
            percls[c].append(d)
            ch.append((abs(d), name_of[c], d))
    ch.sort(reverse=True)
    top = ' '.join('%s%+.2f' % (n[:12], d) for _, n, d in ch[:3])
    print('%-18s %7.3f %7.3f %+8.3f   %s' % (p, bv, bk, bk - bv, top))

ds = np.array(ds)
print()
print('=== 跨港通用性 ===')
print('  平均 ΔBA %+.4f  |  中位 %+.4f  |  正 %d/%d 港' % (ds.mean(), np.median(ds), int((ds > 0).sum()), len(ds)))
print()
print('%-26s %8s %6s' % ('class', 'mean Δ', '同向港数'))
for c in range(C):
    v = np.array(percls[c])
    if len(v):
        print('%-26s %+8.3f  %4d/%d' % (name_of[c], v.mean(), int((v > 0).sum()), len(v)))

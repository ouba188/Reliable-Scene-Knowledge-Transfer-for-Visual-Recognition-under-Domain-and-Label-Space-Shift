"""e16e: CLEAN protocol — alpha chosen per outer fold using source ports only (target fully excluded).

Arms: V | K(0-70) std | K(0-70) std + row-shuffle | K(AIS) std [leaky ref] | K(0-42 facility) std
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
C = int(y.max()) + 1
ALPHAS = [0.1, 0.3, 1.0]
rng = np.random.default_rng(7)
K_shuf = K[rng.permutation(len(ports))]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te, dims, Kd):
    if not dims:
        return X[tr], X[te]
    A, B = Kd[tr][:, dims], Kd[te][:, dims]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def outer(arm_dims, Kd):
    dims = arm_dims
    out = []
    for p in sorted(set(ports.tolist())):
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
            continue
        # 内层：只用非目标港选 alpha
        inner = [q for q in sorted(set(ports[tr].tolist()))][:3]
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
        out.append(ba(y[te], RidgeClassifier(alpha=best_a, class_weight='balanced').fit(ftr, y[tr]).predict(fte)))
    return float(np.mean(out))


LEGAL = list(range(0, 71)); FAC = list(range(0, 42)); AIS = list(range(71, 82))
print('%-30s %8s %9s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 50)
v = outer([], K)
print('%-30s %8.4f %9s' % ('V', v, '—'))
for lab, dims, Kd in [('K(0-70 legal) std', LEGAL, K), ('K(0-70) std ROW-SHUFFLED', LEGAL, K_shuf),
                      ('K(0-42 facility) std', FAC, K), ('K(AIS 71-81) std [LEAKY]', AIS, K)]:
    m = outer(dims, Kd)
    print('%-30s %8.4f %+9.4f' % (lab, m, m - v))

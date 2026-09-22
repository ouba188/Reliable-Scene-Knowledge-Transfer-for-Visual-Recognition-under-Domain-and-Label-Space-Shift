"""e26: prior-shift ALIGNMENT criterion (new construction, not the e23 KL/entropy version).

Insight from our own per-class data: the knowledge acts as a FIXED class-direction prior
(it lifts dry-cargo classes and suppresses tanker/offshore in ~20/24 ports). So its net gain should
depend on whether the target port's class composition is shifted TOWARD the classes the knowledge
raises. Criterion (target side is label-free; the direction is learned on source ports with labels):

    g_hat(p) = < pi_V(p) - mean_{q in source} pi_V(q),  dK >
      pi_V(p) : visual head's mean predicted class distribution on port p
      dK      : knowledge's per-class recall change measured on the SOURCE via inner LOO (8-dim)

Variants: (B) use the K head's own prior shift, (C) cosine instead of inner product,
(D) shuffle control. Reference: always-on, oracle, random.
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
LEGAL = list(range(0, 71))
ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))
rng = np.random.default_rng(23)


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def heads(tr, te):
    A, B = feats(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    return cv.predict(X[te]), ck.predict(B)


def prior(pred):
    h = np.bincount(pred, minlength=C).astype(float)
    return h / max(1, h.sum())


arms = {a: [] for a in ['V', 'K', 'alignV', 'alignK', 'alignCos', 'shuf', 'oracle', 'rand']}
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv_te, pk_te = heads(tr, te)
    pv_tr, pk_tr = heads(tr, tr)
    bv, bk = ba(y[te], pv_te), ba(y[te], pk_te)
    # 源端类方向 dK：内层 LOO（只用源港，目标港完全不参与）
    dK = np.zeros(C); nq = 0
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b = heads(itr, ite)
        for c in range(C):
            m = y[ite] == c
            if m.any():
                dK[c] += float((b[m] == c).mean()) - float((a[m] == c).mean())
        nq += 1
    if nq == 0:
        continue
    dK /= nq
    # 目标港 vs 源港平均的类先验偏移
    src_prior_V = prior(pv_tr); src_prior_K = prior(pk_tr)
    shiftV = prior(pv_te) - src_prior_V
    shiftK = prior(pk_te) - src_prior_K
    gA = float(shiftV @ dK)
    gB = float(shiftK @ dK)
    gC = float(shiftV @ dK / (np.linalg.norm(shiftV) * np.linalg.norm(dK) + 1e-9))
    gS = float(shiftV @ dK[rng.permutation(C)])
    wA, wB, wC, wS = gA > 0, gB > 0, gC > 0, gS > 0
    wO = bk > bv
    wR = rng.random() < 0.5
    arms['V'].append(bv); arms['K'].append(bk)
    arms['alignV'].append(bk if wA else bv); arms['alignK'].append(bk if wB else bv)
    arms['alignCos'].append(bk if wC else bv); arms['shuf'].append(bk if wS else bv)
    arms['oracle'].append(bk if wO else bv); arms['rand'].append(bk if wR else bv)
    tbl.append((p, bv, bk, gA, gB, gC, wA, wO, bk > bv))

b = float(np.mean(arms['V']))
print('%-10s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 30)
for a in ['V', 'K', 'alignV', 'alignK', 'alignCos', 'shuf', 'oracle', 'rand']:
    m = float(np.mean(arms[a]))
    print('%-10s %8.4f %+10.4f' % (a, m, m - b))
okA = sum(1 for t in tbl if t[6] == t[8])
print('\nalignV 方向判对 %d/%d' % (okA, len(tbl)))
print()
print('%-18s %6s %6s %9s %9s %6s' % ('port', 'V', 'K', 'g_alignV', 'g_alignK', 'pick'))
for t in sorted(tbl, key=lambda x: -x[3]):
    print('%-18s %6.3f %6.3f %+9.4f %+9.4f %6s' % (t[0], t[1], t[2], t[3], t[4], 'K' if t[6] else 'V'))
xg = np.array([t[3] for t in tbl]); yg = np.array([t[2] - t[1] for t in tbl])
print('\ncorr(g_alignV, true gain) = %+.3f' % float(np.corrcoef(xg, yg)[0, 1]))

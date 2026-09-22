"""e28: thresholded prior-shift alignment gate (clean).

Criterion:  g = < pi_V(target) - pi_V(source) , dK >,  dK = knowledge's per-class effect on source.
Decision :  use K iff g > tau,  tau chosen on source-side (g_q, gain_q) pairs (nested LOO, target excluded).
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
LEGAL = list(range(0, 71)); ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))
TAUS = [-1e9, 0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015]


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def preds(tr, te):
    A, B = feats(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    return cv.predict(X[te]), ck.predict(B)


def prior(pr):
    h = np.bincount(pr, minlength=C).astype(float)
    return h / max(1, h.sum())


res = {a: [] for a in ['V', 'K', 'tau0', 'tau', 'oracle']}
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv_te, pk_te = preds(tr, te)
    pv_tr, _ = preds(tr, tr)
    bv, bk = ba(y[te], pv_te), ba(y[te], pk_te)
    src_prior = prior(pv_tr)

    qlist, dK_parts, shifts, gains = [], [], [], []
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b = preds(itr, ite)
        atr, btr = preds(itr, itr)
        eff = np.zeros(C)
        for c in range(C):
            m = y[ite] == c
            if m.any():
                eff[c] = float((b[m] == c).mean()) - float((a[m] == c).mean())
        dK_parts.append(eff)
        shifts.append(prior(a) - prior(atr))
        gains.append(ba(y[ite], b) - ba(y[ite], a))
        qlist.append(q)
    if not qlist:
        continue
    dK = np.mean(np.array(dK_parts), 0)
    gq = np.array([s @ dK for s in shifts]); go = np.array(gains)
    gp = float((prior(pv_te) - src_prior) @ dK)
    best_t, best_v = 0.0, -1e9
    for t in TAUS:
        v = float(np.mean(np.where(gq > t, go, 0.0)))
        if v > best_v:
            best_v, best_t = v, t
    res['V'].append(bv); res['K'].append(bk)
    res['tau0'].append(bk if gp > 0 else bv)
    res['tau'].append(bk if gp > best_t else bv)
    res['oracle'].append(bk if bk > bv else bv)
    tbl.append((p, bv, bk, gp, best_t, bk if gp > best_t else bv, bk > bv))

b = float(np.mean(res['V']))
print('%-8s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 28)
for a in ['V', 'K', 'tau0', 'tau', 'oracle']:
    m = float(np.mean(res[a]))
    print('%-8s %8.4f %+10.4f' % (a, m, m - b))
ok0 = sum(1 for t in tbl if (t[3] > 0) == t[6])
okt = sum(1 for t in tbl if (t[3] > t[4]) == t[6])
print('\ntau=0 判对 %d/%d ; 阈值版判对 %d/%d' % (ok0, len(tbl), okt, len(tbl)))
print('\n%-18s %6s %6s %9s %7s %6s %6s' % ('port', 'V', 'K', 'g', 'tau*', 'pick', 'true'))
for t in sorted(tbl, key=lambda x: -x[3]):
    print('%-18s %6.3f %6.3f %+9.4f %7.3f %6s %6s' % (
        t[0], t[1], t[2], t[3], t[4], 'K' if t[3] > t[4] else 'V', 'K' if t[6] else 'V'))

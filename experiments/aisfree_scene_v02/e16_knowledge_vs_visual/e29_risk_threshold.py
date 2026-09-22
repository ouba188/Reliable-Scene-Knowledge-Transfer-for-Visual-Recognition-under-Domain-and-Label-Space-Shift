"""e29: risk-averse threshold for the prior-shift alignment gate (EnsV-style 'avoid the worst case').

Same criterion g = < pi_V(target) - pi_V(source), dK >; decision "use K iff g > tau".
e28 showed the EXPECTED-gain objective degenerates to always-on. Here tau is chosen on the source by
risk-averse objectives:  (a) mean - std   (b) 25th percentile   (c) min  -- plus tau=0 for reference.
Also prints the full source-side tau curve so we can see whether a usable tau exists at all.
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
TAUS = [-1e9, 0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.015]


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


res = {a: [] for a in ['V', 'K', 'tau0', 'risk', 'p25', 'oracle']}
curve = {t: [] for t in TAUS}
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv_te, pk_te = preds(tr, te)
    pv_tr, _ = preds(tr, tr)
    bv, bk = ba(y[te], pv_te), ba(y[te], pk_te)
    src_prior = prior(pv_tr)

    parts, shifts, gains = [], [], []
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b = preds(itr, ite)
        atr, _ = preds(itr, itr)
        eff = np.zeros(C)
        for c in range(C):
            m = y[ite] == c
            if m.any():
                eff[c] = float((b[m] == c).mean()) - float((a[m] == c).mean())
        parts.append(eff); shifts.append(prior(a) - prior(atr)); gains.append(ba(y[ite], b) - ba(y[ite], a))
    if not parts:
        continue
    dK = np.mean(np.array(parts), 0)
    gq, go = np.array([s @ dK for s in shifts]), np.array(gains)

    def val(t):
        return np.where(gq > t, go, 0.0)

    stats = {t: val(t) for t in TAUS}
    for t in TAUS:
        curve[t].append(stats[t])
    pick_risk = max(TAUS, key=lambda t: float(np.mean(stats[t]) - np.std(stats[t])))
    pick_p25 = max(TAUS, key=lambda t: float(np.percentile(stats[t], 25)))
    gp = float((prior(pv_te) - src_prior) @ dK)
    res['V'].append(bv); res['K'].append(bk)
    res['tau0'].append(bk if gp > 0 else bv)
    res['risk'].append(bk if gp > pick_risk else bv)
    res['p25'].append(bk if gp > pick_p25 else bv)
    res['oracle'].append(bk if bk > bv else bv)
    tbl.append((p, bv, bk, gp, pick_risk, pick_p25,
                bk if gp > pick_risk else bv, bk if gp > pick_p25 else bv, bk > bv))

b = float(np.mean(res['V']))
print('%-8s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 28)
for a in ['V', 'K', 'tau0', 'risk', 'p25', 'oracle']:
    m = float(np.mean(res[a]))
    print('%-8s %8.4f %+10.4f' % (a, m, m - b))
ok0 = sum(1 for t in tbl if (t[3] > 0) == t[8])
okr = sum(1 for t in tbl if (t[3] > t[4]) == t[8])
okp = sum(1 for t in tbl if (t[3] > t[5]) == t[8])
print('\ntau0 判对 %d/%d ; risk 判对 %d/%d ; p25 判对 %d/%d' % (ok0, len(tbl), okr, len(tbl), okp, len(tbl)))
print()
print('=== 源端 tau 曲线（跨折平均）：mean gain / std / p25 / 应用比例 ===')
for t in TAUS:
    arr = np.concatenate([a for a in curve[t]]) if curve[t] else np.array([0.0])
    applied = np.mean([float(np.mean(a != 0)) for a in curve[t]])
    print('  tau=%8.3f  mean %+.4f  std %.4f  p25 %+.4f  应用率 %.2f' % (
        t if t > -1e8 else -999, float(arr.mean()), float(arr.std()),
        float(np.percentile(arr, 25)), applied))
print()
print('%-18s %6s %6s %9s %8s %8s %6s %6s' % ('port', 'V', 'K', 'g', 'tau_risk', 'tau_p25', 'pick', 'true'))
for t in sorted(tbl, key=lambda x: -x[3]):
    print('%-18s %6.3f %6.3f %+9.4f %8.3f %8.3f %6s %6s' % (
        t[0], t[1], t[2], t[3], t[4], t[5], 'K' if t[3] > t[4] else 'V', 'K' if t[8] else 'V'))

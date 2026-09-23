"""e42: turn the G predictor into an actual gate and measure end-to-end BA.

Per fold: fit the G predictor on the SOURCE ports (nested LOO), tune the gate threshold tau on the
source ports' own gated BA, then on the held-out target predict V, or V+K per instance depending on
p(rescue) > tau. Compare BA: V | K always | gated(+K) | gated(oracle, uses target labels).

The instant-level gate is new: E21 only tested a PORT-level gate.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0]
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PORT_U = sorted(set(ports.tolist()))
NSRC = 8


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fit_arm(tr, te, alpha):
    A, B = kstd(tr, te)
    pv = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(X[tr], y[tr]).predict(X[te])
    pk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(
        np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    sv = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(X[tr], y[tr]).decision_function(X[te])
    sk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(A, y[tr]).decision_function(B)
    return pv, pk, sv, sk, B


def gfeats(sv, sk, Ks, pred_v):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    margk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pred_v).astype(float)[:, None]
    oh = np.zeros((len(pred_v), C - 1)); oh[np.arange(len(pred_v)), np.minimum(pred_v, C - 2)] = 1
    return np.c_[marg, ent, margk, dis, oh, Ks]


def tune_alpha(src):
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in src[:3]:
            tr = np.where(np.isin(ports, src) & (ports != q))[0]; te = np.where(ports == q)[0]
            if len(tr) < 200 or len(set(y[tr].tolist())) < C:
                continue
            sc.append(ba(y[te], fit_arm(tr, te, a)[1]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


print('%-20s %7s %7s %7s %7s %6s' % ('target port', 'V', 'K_all', 'K_gate', 'K_orac', 'tau'))
print('-' * 62)
res = []
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    a = tune_alpha(src)
    Gtr, Ftr, cor = [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200 or len(set(y[trq].tolist())) < C:
            continue
        pv, pk, sv, sk, Ks = fit_arm(trq, teq, a)
        g = np.zeros(len(teq)); g[(pv != y[teq]) & (pk == y[teq])] = 1; g[(pv == y[teq]) & (pk != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        Gtr.append(g[m]); Ftr.append(gfeats(sv[m], sk[m], Ks[m], pv[m]))
        cor.append(np.c_[(pv == y[teq])[m], (pk == y[teq])[m], y[teq][m]])
    if not Gtr:
        continue
    Gtr = np.concatenate(Gtr); Ftr = np.vstack(Ftr); cor = np.vstack(cor)
    ytr = (Gtr > 0).astype(int)
    gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(Ftr, ytr)
    ps = gbm.predict_proba(Ftr)[:, 1]
    best_t, best = 0.5, -1
    for t in TAUS:                        # tau tuned on the SOURCE ports' gated BA
        sel = ps > t
        ok = np.where(sel, cor[:, 1], cor[:, 0]).astype(bool)
        sc = [float(ok[cor[:, 2] == c].mean()) for c in range(C) if (cor[:, 2] == c).sum() > 0]
        if sc and float(np.mean(sc)) > best:
            best, best_t = float(np.mean(sc)), t
    tr = np.where(ports != p)[0]
    pv, pk, sv, sk, Ks = fit_arm(tr, te, a)
    Ft = gfeats(sv, sk, Ks, pv)
    pt = gbm.predict_proba(Ft)[:, 1]
    v = ba(y[te], pv); kk = ba(y[te], pk)
    gate = np.where(pt > best_t, pk, pv)
    oracle = np.where((pk != y[te]) & (pv == y[te]), pv, pk)
    res.append((p, v, kk, ba(y[te], gate), ba(y[te], oracle), best_t))
    print('%-20s %7.3f %7.3f %7.3f %7.3f %6.2f' % res[-1])

print()
for i, lab in [(1, 'V'), (2, 'K 全用'), (3, 'K 闸门'), (4, 'K oracle')]:
    d = np.array([r[i] for r in res]) - np.array([r[1] for r in res])
    m = np.array([r[i] for r in res])
    print('%-10s mean BA %.4f  Δ %+7.4f  逐港正 %2d/%d  worst %+.2f' % (
        lab, m.mean(), d.mean(), (d > 0).sum(), len(d), d.min() * 100))
print()
print('tau 取值:', {t: sum(1 for r in res if abs(r[5] - t) < 1e-9) for t in sorted({r[5] for r in res})})

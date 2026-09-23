"""e45 (b): within-port percentile knowledge -- change the COORDINATE, not the subset.

r[i,d] = percentile rank of K[i,d] among the chips of the SAME port (label-free: only uses the port
assignment + the knowledge values). This removes the port's *scale* while keeping the relative
semantics -- the only intervention so far that targets the r=0.771 entanglement itself.

Reports: (1) the I/class-vs-I/port audit for K vs K_pct, (2) the usual LOO arms,
(3) the gate from e42 on the percentile features.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax
from scipy import stats

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
PORT_U = sorted(set(ports.tolist())); pid = np.array([PORT_U.index(p) for p in ports])
NSRC = 8

# ---- within-port percentile transform -------------------------------------
Kp = np.zeros_like(K)
for j in range(len(PORT_U)):
    m = pid == j
    sub = K[m][:, LEGAL]
    n = sub.shape[0]
    order = sub.argsort(0).argsort(0).astype(np.float64)     # rank per dim
    Kp[np.ix_(m, LEGAL)] = order / max(1, n - 1)
KB = {'raw': K, 'pct': Kp}

NB = 16


def mi(x, lab, nlab):
    ok = np.isfinite(x); x, lab = x[ok], lab[ok]
    if x.std() < 1e-12:
        return 0.0
    e = np.unique(np.quantile(x, np.linspace(0, 1, NB + 1)))
    if len(e) < 3:
        return 0.0
    b = np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2)
    j = np.zeros((len(e) - 1, nlab))
    for bi, li in zip(b, lab):
        j[bi, li] += 1
    p = j / j.sum(); px = p.sum(1, keepdims=True); pl = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ pl)[nz])).sum())


cnt = np.bincount(pid, minlength=len(PORT_U)) / len(pid)
hp = -float((cnt[cnt > 0] * np.log(cnt[cnt > 0])).sum())
cntc = np.bincount(y, minlength=C) / len(y)
hc = -float((cntc[cntc > 0] * np.log(cntc[cntc > 0])).sum())
print('%-8s %10s %10s %8s' % ('version', 'mean I/port', 'mean I/class', 'corr'))
for nm, kk in KB.items():
    ip = np.array([mi(kk[:, i], pid, len(PORT_U)) / hp for i in LEGAL])
    ic = np.array([mi(kk[:, i], y, C) / hc for i in LEGAL])
    print('%-8s %10.4f %10.4f %8.3f' % (nm, ip.mean(), ic.mean(), np.corrcoef(ic, ip)[0, 1]))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te, Kd):
    A, B = Kd[tr][:, LEGAL], Kd[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fit_arm(tr, te, alpha, Kd, cache=None):
    Ka, Kb = cache if cache is not None else prep(tr, te, Kd)
    clfv = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(X[tr], y[tr])
    pv = clfv.predict(X[te]); sv = clfv.decision_function(X[te])
    sk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(Ka, y[tr]).decision_function(Kb)
    pk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(np.c_[X[tr], Ka], y[tr]).predict(np.c_[X[te], Kb])
    return pv, pk, sv, sk, Kb


def f_cur(sv, sk, Ks, pv):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    mk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk, dis, oh, Ks]


def tune_alpha(src, Kd):
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in src[:3]:
            tr = np.where(np.isin(ports, src) & (ports != q))[0]; te = np.where(ports == q)[0]
            if len(tr) < 200 or len(set(y[tr].tolist())) < C:
                continue
            sc.append(ba(y[te], fit_arm(tr, te, a, Kd)[1]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


print()
print('%-20s %7s %7s %7s %7s %7s' % ('port', 'V', 'K_raw', 'K_pct', 'gate_pct', 'AUC'))
print('-' * 62)
res = []
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    ar = tune_alpha(src, K); ap = tune_alpha(src, Kp)
    tr = np.where(ports != p)[0]
    pv, pkr, _, _, _ = fit_arm(tr, te, ar, K)
    _, pkp, _, _, _ = fit_arm(tr, te, ap, Kp)
    F1, G, CV, CK, YV = [], [], [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200 or len(set(y[trq].tolist())) < C:
            continue
        pvq, pkq, svq, skq, Ksq = fit_arm(trq, teq, ap, Kp)
        g = np.zeros(len(teq)); g[(pvq != y[teq]) & (pkq == y[teq])] = 1; g[(pvq == y[teq]) & (pkq != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        F1.append(f_cur(svq[m], skq[m], Ksq[m], pvq[m])); G.append(g[m])
        CV.append((pvq == y[teq])[m]); CK.append((pkq == y[teq])[m]); YV.append(y[teq][m])
    gate = float('nan'); auc = float('nan'); t = 0.5
    if G:
        G = np.concatenate(G); CV = np.concatenate(CV); CK = np.concatenate(CK); YV = np.concatenate(YV)
        ytr = (G > 0).astype(int)
        if ytr.min() != ytr.max():
            gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(np.vstack(F1), ytr)
            ps = gbm.predict_proba(np.vstack(F1))[:, 1]
            def gacc(sel):
                ok = np.where(sel, CK, CV).astype(bool)
                per = [float(ok[YV == c].mean()) for c in range(C) if (YV == c).sum() > 0]
                return float(np.mean(per)) if per else -1
            t = max(TAUS, key=lambda tt: gacc(ps > tt))
            _, _, sv, sk, Ks = fit_arm(tr, te, ap, Kp)
            b = f_cur(sv, sk, Ks, pv)
            pt = gbm.predict_proba(b)[:, 1]
            gate = ba(y[te], np.where(pt > t, pkp, pv))
            g = np.zeros(len(te)); g[(pv != y[te]) & (pkp == y[te])] = 1; g[(pv == y[te]) & (pkp != y[te])] = -1
            m = g != 0
            if m.sum() > 10 and len(set((g[m] > 0).tolist())) > 1:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score((g[m] > 0).astype(int), pt[m])
    res.append((p, ba(y[te], pv), ba(y[te], pkr), ba(y[te], pkp), gate, auc))
    print('%-20s %7.3f %7.3f %7.3f %7.3f %7.3f' % res[-1], flush=True)

V = np.array([r[1] for r in res]); KR = np.array([r[2] for r in res])
KP = np.array([r[3] for r in res]); GT = np.array([r[4] for r in res])
print()
for lab, m in [('V', V), ('K 原始', KR), ('K 港内百分位', KP), ('K 百分位+闸门', GT)]:
    d = (m - V) * 100
    d = d[np.isfinite(d)]
    print('%-14s mean %.4f  Δ %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (lab, np.nanmean(m), d.mean(), (d > 0).sum(), len(d), d.min()))
d1 = (KP - V) * 100; d2 = (KR - V) * 100
p1 = d1[np.isfinite(d1)]; p2 = d2[np.isfinite(d2)]
print()
print('配对（原始 vs 百分位）: 差 %+.2f pp  Wilcoxon p=%.4f' % ((p1 - p2).mean(), stats.wilcoxon(p1, p2).pvalue))
av = np.array([r[5] for r in res]); av = av[np.isfinite(av)]
if len(av):
    print('百分位版 G 预测 AUC: 平均 %.3f  >0.6 的港 %d/%d' % (av.mean(), (av > 0.6).sum(), len(av)))

"""e68 = e67 (BBSE) + a support-set mask from the estimated marginal, and per-fold dumping.

Partial-domain fix: the target port holds only 3-6 of the 8 classes, so reweighting over all 8 pushes mass
onto classes that cannot exist there and predictions collapse to a constant. The estimated p_t(y) itself
identifies the support set (classes with negligible estimated mass), so zero those posteriors before the
argmax -- still 100% label-free. tau is swept; selecting tau on source ports is left as the deployment rule.

Dumps e68_preds.pkl (per fold: y_t, posteriors, estimated marginal, raw histogram) so later variants cost
no refit.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz')['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _pj in PORT_U:
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)

SUBS = 20000
NCM = 10
TAUS = [0.005, 0.01, 0.02, 0.05]
rng = np.random.default_rng(0)


def std_k(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu, s = A.mean(0), A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def fit_predict(tr, te, use_k, ytr):
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    A, B = std_k(trs, te)
    M = np.c_[X[trs], A] if use_k else X[trs]
    Q = np.c_[X[te], B] if use_k else X[te]
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(M, ytr[trs])
    P = softmax(rv.decision_function(Q))
    cls = rv.classes_
    Pfull = np.zeros((len(te), C)); Pfull[:, cls] = P
    return Pfull, cls[P.argmax(1)]


def bbse(phat, Cm):
    Cm = Cm + 1e-2 * np.eye(C)
    Cm = Cm / np.maximum(Cm.sum(0, keepdims=True), 1e-9)
    p, _ = nnls(Cm.T, phat)
    s = p.sum()
    return p / s if s > 1e-9 else np.full(C, 1.0 / C)


def confmat(y_true, y_pred):
    M = np.zeros((C, C))
    for t_, p_ in zip(y_true, y_pred):
        M[p_, t_] += 1
    return M / np.maximum(M.sum(0, keepdims=True), 1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


arms = ['V', 'VK', 'VK_bbse'] + ['VK_bbse_mask%.3f' % t for t in TAUS] + ['VK_bbse_shuf']
res = {a: [] for a in arms}
dump = []
print('%-18s %7s %8s %9s %s' % ('port', 'V', 'VK', 'VK_bbse', ' '.join('%9s' % ('mask%.3f' % t) for t in TAUS)), flush=True)

for p in PORT_U:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    others = [x for x in PORT_U if x != p][:NCM]
    Cmats = {nm: [] for nm in ['V', 'VK']}
    for q in others:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        for nm, use_k in [('V', False), ('VK', True)]:
            _, yp = fit_predict(trq, teq, use_k, y)
            Cmats[nm].append(confmat(y[teq], yp))
    ps = np.bincount(y[tr], minlength=C) / len(tr)
    true = np.bincount(y[te], minlength=C) / len(te)
    out = {}
    for nm, use_k in [('V', False), ('VK', True)]:
        P_t, yp_t = fit_predict(tr, te, use_k, y)
        out[nm] = (P_t, yp_t)
    Cm = np.mean(Cmats['VK'], 0)
    ph = np.bincount(out['VK'][1], minlength=C) / len(te)
    pt_hat = bbse(ph, Cm)
    w = np.clip(pt_hat / np.maximum(ps, 1e-6), 1e-6, None)
    Pc = out['VK'][0] * w; Pc = Pc / np.maximum(Pc.sum(1, keepdims=True), 1e-12)
    out['VK_bbse'] = (Pc, Pc.argmax(1))
    for t in TAUS:
        keep = (pt_hat >= t).astype(float)
        Pk = Pc * keep
        rowsum = Pk.sum(1, keepdims=True)
        Pk = np.where(rowsum > 1e-12, Pk / np.maximum(rowsum, 1e-12), out['VK'][0])   # unmasked if nothing survives
        out['VK_bbse_mask%.3f' % t] = (Pk, Pk.argmax(1))
    ysh = y.copy(); perm = rng.permutation(len(tr)); ysh[tr] = y[tr][perm]
    P_s, yp_s = fit_predict(tr, te, True, ysh)
    Cs = []
    for q in others:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        _, yq = fit_predict(trq, teq, True, ysh)
        Cs.append(confmat(ysh[teq], yq))
    ptS = bbse(np.bincount(yp_s, minlength=C) / len(te), np.mean(Cs, 0))
    wS = np.clip(ptS / np.maximum(ps, 1e-6), 1e-6, None)
    PcS = P_s * wS; PcS = PcS / np.maximum(PcS.sum(1, keepdims=True), 1e-12)
    out['VK_bbse_shuf'] = (PcS, PcS.argmax(1))
    for a in arms:
        res[a].append(ba(y[te], out[a][1]))
    dump.append({'port': p, 'y': y[te].copy(), 'ps': ps, 'pt_hat': pt_hat, 'ph': ph, 'true': true,
                 'post': {a: out[a][0] for a in arms if a != 'V'}, 'pred': {a: out[a][1] for a in arms}})
    print('%-18s %7.3f %8.3f %9.3f %s' % (p, res['V'][-1], res['VK'][-1], res['VK_bbse'][-1],
                                          ' '.join('%9.3f' % res['VK_bbse_mask%.3f' % t][-1] for t in TAUS)), flush=True)

pickle.dump(dump, open('e68_preds.pkl', 'wb'))     # our own artifact; later variants need no refit
print()
for a in arms:
    v = np.array(res[a]); d = (v - np.array(res['V'])) * 100
    print('%-20s mean %.4f  Δ vs ridge_V %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        a, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
print()
tv = [0.5 * np.abs(dd['pt_hat'] - dd['true']).sum() for dd in dump]
tv0 = [0.5 * np.abs(dd['ph'] - dd['true']).sum() for dd in dump]
print('TV(est,true) 中位 %.3f ｜ 原始直方图 %.3f ｜ 支持集恢复率（真类被掩住的比例）:' % (np.median(tv), np.median(tv0)))
for t in TAUS:
    miss = np.mean([float((~((dd['pt_hat'] >= t) & (dd['true'] > 0)) & (dd['true'] > 0)).sum()) / max(1, (dd['true'] > 0).sum()) for dd in dump])
    extra = np.mean([float(((dd['pt_hat'] >= t) & (dd['true'] == 0)).sum()) / max(1, (dd['true'] == 0).sum()) for dd in dump])
    print('   tau=%.3f: 真类被误掩 %.1f%%   空类被误留 %.1f%%' % (t, miss * 100, extra * 100))

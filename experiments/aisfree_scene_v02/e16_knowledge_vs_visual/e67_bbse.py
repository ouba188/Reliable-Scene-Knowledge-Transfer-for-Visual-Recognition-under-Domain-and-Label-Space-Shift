"""e67: label-shift correction (BBSE) using ONLY label-free target information, plus the three safeguards.

Deployable pipeline per fold (target port p held out):
  (i)   fit the base classifiers on the source ports;
  (ii)  predict the target's UNLABELLED chips -> posterior P_t and hard labels yhat_t; the histogram
        phat_t(yhat) is therefore a function of (unlabelled target, source labels) only;
  (iii) build the source confusion matrix C from leave-one-source-port-out predictions on SOURCE ports
        (NCM ports -- 3 is not enough: individual ports hold only 3-6 classes, which makes C rank-deficient
        and the NNLS solution degenerate; that failure was observed and is why NCM=10 + a ridge term);
  (iv)  solve C^T p = phat_t for p_t(y) (NNLS + ridge + normalise) -> w = p_t_hat / p_s -> P_t * w -> argmax.
No target label is read in (ii)-(iv). Target labels are used only afterwards to score BA and to measure the
estimator's own error (the sole legitimate use of the true marginal).
"""
import csv
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
NCM = 10            # source ports feeding the confusion matrix (rank matters, see docstring)
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
    if not hasattr(bbse, 'r'):
        bbse.r = float(np.linalg.cond(Cm))
        print('  [confusion] cond=%.1f  rank=%d' % (bbse.r, np.linalg.matrix_rank(Cm)), flush=True)
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


res = {a: [] for a in ['V', 'VK', 'V_bbse', 'VK_bbse', 'VK_bbse_shuf']}
tv, tv_shuf, tv01 = [], [], []
print('%-18s %7s %8s %9s %10s %10s   %s' % ('port', 'V', 'VK', 'V+BBSE', 'VK+BBSE', 'VK+BBSE_shuf', 'TV(est,true)'), flush=True)

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
    out = {}
    for nm, use_k in [('V', False), ('VK', True)]:
        P_t, yp_t = fit_predict(tr, te, use_k, y)
        out[nm] = (P_t, yp_t)
        Cm = np.mean(Cmats[nm], 0)
        ph = np.bincount(yp_t, minlength=C) / len(te)
        pt_hat = bbse(ph, Cm)
        w = np.clip(pt_hat / np.maximum(ps, 1e-6), 1e-6, None)
        Pc = P_t * w; Pc = Pc / np.maximum(Pc.sum(1, keepdims=True), 1e-12)
        out[nm + '_bbse'] = (Pc, Pc.argmax(1))
        if nm == 'VK':
            true = np.bincount(y[te], minlength=C) / len(te)
            tv.append(0.5 * np.abs(pt_hat - true).sum())
            tv01.append(0.5 * np.abs(ph - true).sum())
    ysh = y.copy(); perm = rng.permutation(len(tr)); ysh[tr] = y[tr][perm]   # permute positions, not index values
    P_t, yp_t = fit_predict(tr, te, True, ysh)
    Cs = []
    for q in others:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        _, yq = fit_predict(trq, teq, True, ysh)
        Cs.append(confmat(ysh[teq], yq))
    CmS = np.mean(Cs, 0)
    ptS = bbse(np.bincount(yp_t, minlength=C) / len(te), CmS)
    true = np.bincount(y[te], minlength=C) / len(te)
    tv_shuf.append(0.5 * np.abs(ptS - true).sum())
    wS = np.clip(ptS / np.maximum(ps, 1e-6), 1e-6, None)
    PcS = P_t * wS; PcS = PcS / np.maximum(PcS.sum(1, keepdims=True), 1e-12)
    out['VK_bbse_shuf'] = (PcS, PcS.argmax(1))
    for a in ['V', 'VK', 'V_bbse', 'VK_bbse', 'VK_bbse_shuf']:
        res[a].append(ba(y[te], out[a][1]))
    print('%-18s %7.3f %8.3f %9.3f %10.3f %10.3f   %.3f' % (
        p, res['V'][-1], res['VK'][-1], res['V_bbse'][-1], res['VK_bbse'][-1], res['VK_bbse_shuf'][-1], tv[-1]), flush=True)

print()
for a, lab in [('V', 'ridge_V'), ('VK', 'ridge_VK'), ('V_bbse', 'ridge_V + BBSE'), ('VK_bbse', 'ridge_VK + BBSE'),
               ('VK_bbse_shuf', 'ridge_VK + BBSE, source labels shuffled (control)')]:
    v = np.array(res[a]); d = (v - np.array(res['V'])) * 100
    print('%-46s mean %.4f  Δ vs ridge_V %+6.2f  pos %2d/%d  worst %+6.2f' % (
        lab, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
print()
print('[evidence 2] TV(p_t_hat, p_t_true): median %.3f mean %.3f   |   raw prediction histogram TV: median %.3f mean %.3f'
      % (np.median(tv), np.mean(tv), np.median(tv01), np.mean(tv01)))
print('[evidence 3] shuffled-source-label control TV: median %.3f mean %.3f  (must be clearly worse)'
      % (np.median(tv_shuf), np.mean(tv_shuf)))
print('   folds where the estimate beats the raw histogram: %d/%d' % (int(sum(1 for a, b in zip(tv, tv01) if a < b)), len(tv)))

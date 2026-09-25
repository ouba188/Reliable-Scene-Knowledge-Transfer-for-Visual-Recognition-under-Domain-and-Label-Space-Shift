"""e141: B -- reopen the prior-correction line with a STABILISED estimator.

The attribution (E18z) says the entire cross-port drop is a class-mix effect: source conditionals under the target priors give
0.3765 against 0.5044 on the source, i.e. a -12.79 pp label-shift component, while the target's own conditionals are +6.54 pp
better. Our earlier BBSE attempt failed, and the attribution says why: the estimator (it was 0.012 better than the raw
prediction histogram), not the shift structure.

Estimator here (label-free, target labels only for scoring):
  - confusion matrix from HELD-OUT SOURCE ports, averaged over 2 inner folds, row-normalised;
  - shrink it toward the identity, C_s = (1-lam) C + lam I, with lam chosen on source-side episodes (no target data);
  - solve min ||C_s^T pi - p_pred||^2 with pi >= 0, sum pi = 1 by non-negative least squares + renormalisation;
  - correct posteriors by the ratio pi_hat / pi_source and re-argmax.

Pre-registered: the correction must recover >= 50% of the -12.79 pp label-shift component
(i.e. (BA_after - BA_before) >= 0.5 * 12.79 pp) with paired p < 0.05 over the 24 ports.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import nnls
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
PU = sorted(set(ports.tolist()))
LS_COMPONENT = 12.79
LAMS = (0.0, 0.05, 0.1, 0.2, 0.4)


def fit(trs, te):
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mu) / sd, y[trs])
    L = m.decision_function((X[te] - mu) / sd)
    if L.shape[1] != C:
        Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
    return L


def conf(pred, true):
    M = np.zeros((C, C))
    for t in range(C):
        m = true == t
        if m.sum():
            for k in range(C):
                M[t, k] = (pred[m] == k).mean()
    return M


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


def prior_hat(Ch, p_pred, lam):
    Cs = (1 - lam) * Ch + lam * np.eye(C)
    A = Cs.T
    pi, _ = nnls(A, p_pred, maxiter=200)
    return pi / max(pi.sum(), 1e-9)


before, after, prior_err = [], [], []
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    Ls = fit(src, te)
    Ps = softmax(Ls, 1); pred = Ps.argmax(1)
    a0 = ba(y[te], pred)
    # source-side confusion from TWO inner folds, averaged (the stabilisation), lam picked on the source side
    Cs, lam_best, lam_score = [], None, -1e9
    inner = [x for x in PU if x != p][:2]
    for q in inner:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 500 or len(teq) < 50:
            continue
        Lq = fit(trq, teq); Cs.append(conf(Lq.argmax(1), y[teq]))
    if not Cs:
        continue
    Ch = np.mean(Cs, 0)
    for lam in LAMS:
        sc = 0.0
        for i, q in enumerate(inner):
            if i >= len(Cs):
                break
            srcq = np.where((ports != p) & (ports != q))[0]
            teq = np.where(ports == q)[0]
            Lq = fit(srcq, teq)
            pq = softmax(Lq, 1)
            h = np.bincount(pq.argmax(1), minlength=C).astype(float); h /= h.sum()
            pi = prior_hat(Ch, h, lam)
            pi_s = np.array([(y[srcq] == c).mean() for c in range(C)])
            w = pi / np.maximum(pi_s, 1e-6)
            sc += ba(y[teq], (pq * w).argmax(1))
        if sc > lam_score:
            lam_score, lam_best = sc, lam
    h_t = np.bincount(pred, minlength=C).astype(float); h_t /= h_t.sum()
    pi_t = prior_hat(Ch, h_t, lam_best)
    pi_s = np.array([(y[src] == c).mean() for c in range(C)])
    w = pi_t / np.maximum(pi_s, 1e-6)
    a1 = ba(y[te], (Ps * w).argmax(1))
    before.append(a0); after.append(a1)
    true_pi = np.array([(y[te] == c).mean() for c in range(C)])
    prior_err.append(0.5 * float(np.abs(pi_t - true_pi).sum()))
    print('%-16s BA %.4f → %.4f (%+.2fpp) ｜ λ*=%.2f ｜ 先验 TV 误差 %.3f' % (p, a0, a1, (a1 - a0) * 100, lam_best, prior_err[-1]), flush=True)

b, a = np.array(before), np.array(after)
d = (a - b) * 100
print('')
print('校正前 %.4f → 校正后 %.4f ｜ ΔBA 中位 %+.2fpp ｜ 正港 %d/%d' % (b.mean(), a.mean(), np.median(d), int((d > 0).sum()), len(d)))
print('先验估计误差（TV）中位 %.3f' % np.median(prior_err))
pv = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
print('配对 p=%.4f' % pv)
rec = 100 * d.mean() / LS_COMPONENT
print('收复先验分量的 %.1f%%（判据要求 ≥50%%）' % rec)
print('')
print('预注册判据（收复 ≥50%% 且 p<0.05）: %s' % ('成立 ✓✓ —— 稳定估计器复开了先验校正 ✓' if (rec >= 50 and pv < 0.05) else '未成立 ✗'))

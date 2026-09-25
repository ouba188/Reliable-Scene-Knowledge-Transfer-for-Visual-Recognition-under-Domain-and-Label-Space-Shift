"""e123: surrogate risk control -- can a LABEL-FREE surrogate carry the guarantee the true risk cannot?

E18n closed the direct route: the true risk cannot be verified at the target port without target labels, and importance
weighting left it at ~2.5 alpha. This tests the remaining move: control a label-free surrogate of the risk at calibration
time, and check whether controlling the surrogate also controls the TRUE risk at the target port.

Surrogates (all computable without any label, on source and target alike):
  sur_conf  = 1 - max_softmax            (confidence deficit)
  sur_dom   = p(target | x)              (domain-classifier discrepancy)
  sur_prod  = sur_conf * sur_dom

Procedure per (target port, alpha): on 2 held-out source ports, accept the largest top-score prefix whose MEAN SURROGATE
stays <= alpha; apply that threshold to the target port and measure the true error rate among accepted instances.
Reference arm: calibrate on the TRUE risk instead (what e121 did, currently holding on 0/23 ports).

Pre-registered: with the best surrogate, true risk <= alpha on >= 80% of the 23 ports.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
X = np.load(ROOT / 'features_all' / 'pca512_all.npy').astype(np.float32)
rows = list(csv.DictReader((ROOT / 'dataset_all' / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
y = np.full(len(rows), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
known = y >= 0
ALPHAS = (0.05, 0.10)
SURS = ('true', 'conf', 'dom', 'prod')


def fit(trs, te):
    Z = X[trs]
    mu = Z.mean(0, keepdims=True); C = ((Z - mu).T @ (Z - mu)) / len(Z)
    C = 0.95 * C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C.astype(np.float64)).astype(np.float32)
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[trs])
    Q = X[te]
    L = clf.decision_function(Q)
    S = softmax(L, 1)
    s = S.max(1)
    # distance-based confidence deficit: how far the feature sits from its predicted class mean
    means = np.stack([Z[y[trs] == j].mean(0) for j in range(8)])
    pr = L.argmax(1)
    d = np.array([(Q[i] - means[pr[i]]) @ P @ (Q[i] - means[pr[i]]) for i in range(len(Q))])
    return s, d, pr


def thr_sur(s, sur, alpha):
    """largest top-score prefix whose mean surrogate stays <= alpha."""
    o = np.argsort(-s); s, sur = s[o], sur[o]
    cum = np.cumsum(sur) / (np.arange(len(sur)) + 1)
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


PU = sorted(set(ports.tolist()))
held = {sur: {a: [] for a in ALPHAS} for sur in SURS}
covr = {sur: {a: [] for a in ALPHAS} for sur in SURS}
for p in PU:
    src = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(src) < 3000 or len(te) < 100:
        continue
    cs, cd, cp, ct, ci = [], [], [], [], []
    for q in [x for x in PU if x != p][:2]:
        trq = np.where(known & (ports != p) & (ports != q))[0]
        teq = np.where(known & (ports == q))[0]
        if len(trq) < 3000 or len(teq) < 50:
            continue
        s, d, pr = fit(trq, teq)
        cs.append(s); cd.append(d); cp.append(pr); ct.append(y[teq]); ci.append(teq)
    if not cs:
        continue
    cs = np.concatenate(cs); cd = np.concatenate(cd); cp = np.concatenate(cp); ct = np.concatenate(ct)
    cidx = np.concatenate(ci)
    lg = LogisticRegression(max_iter=1000).fit(np.vstack([X[cidx], X[te]]), np.r_[np.zeros(len(cidx)), np.ones(len(te))])
    cdom = lg.predict_proba(X[cidx])[:, 1]
    # normalise surrogates to the same scale so a common alpha is meaningful: divide by their calibration mean
    sur = {'true': (cp != ct).astype(float),
           'conf': 1 - cs, 'dom': cdom, 'prod': (1 - cs) * cdom}
    for k in sur:
        sur[k] = sur[k] / max(sur[k].mean(), 1e-6)
    s_t, d_t, pr_t = fit(src, te)
    dom_t = lg.predict_proba(X[te])[:, 1]
    sur_t = {'true': None, 'conf': 1 - s_t, 'dom': dom_t, 'prod': (1 - s_t) * dom_t}
    ok = (pr_t == y[te])
    for a in ALPHAS:
        for k in SURS:
            t = thr_sur(cs, sur[k], a)
            acc = s_t - t >= 0
            held[k][a].append(bool(acc.sum() >= 20 and (~ok[acc]).mean() <= a + 1e-9))
            covr[k][a].append(float(acc.mean()))
    print('%-16s done (%d/%d)' % (p, len(held['conf'][0.05]), len(PU)), flush=True)

print('')
print('=== 真风险 <= α 的港数（校准信号 → 24 港）===')
print('%-8s %6s %14s %12s %10s' % ('校准信号', 'alpha', '守约港数/23', '守约率', '中位覆盖率'))
for a in ALPHAS:
    for k in SURS:
        h = held[k][a]
        print('%-8s %6.2f %14s %11.0f%% %10.3f' % (k, a, '%d/%d' % (sum(h), len(h)), 100 * sum(h) / len(h),
                                                   float(np.median(covr[k][a]))))
print('')
best = max(SURS, key=lambda k: sum(held[k][0.05]))
print('预注册判据（最佳代理 真风险<=α 的港 >=80%%，α=0.05）: 最佳代理=%s，%.0f%% ⇒ %s'
      % (best, 100 * sum(held[best][0.05]) / len(held[best][0.05]),
         '成立 ✓✓' if sum(held[best][0.05]) >= 0.8 * len(held[best][0.05]) else '未成立 ✗'))

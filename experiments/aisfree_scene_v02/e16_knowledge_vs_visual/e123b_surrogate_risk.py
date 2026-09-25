"""e123b: surrogate risk control, done properly -- fit surrogate -> error, then control the PREDICTED risk.

e123 v1 was ill-posed: it normalised each surrogate by its calibration mean and required the MEAN surrogate to stay <= alpha,
i.e. <= 5% of its own mean, so the accepted set collapsed to empty (coverage 0.000) and the hypothesis was never tested. The
meaningful construction maps the surrogate to a probability of error first:

  on the calibration ports, fit g: (acceptance score, label-free surrogates) -> P(error), using calibration labels only;
  accept the largest top-score prefix whose MEAN PREDICTED risk <= alpha;
  then measure the TRUE error rate among accepted instances at the target port.

The diagnostic that decides whether this can work at all: the AUC of g on the TARGET port. If a source-fit error model does
not rank the target's errors, no threshold on it can control the target's risk. Labels are used for evaluation only.

Pre-registered: true risk <= alpha on >= 80% of the 23 ports, alpha=0.05.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score

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


def fit(trs, te):
    Z = X[trs]
    mu = Z.mean(0, keepdims=True); C = ((Z - mu).T @ (Z - mu)) / len(Z)
    C = 0.95 * C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C.astype(np.float64)).astype(np.float32)
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[trs])
    Q = X[te]
    L = clf.decision_function(Q)
    return softmax(L, 1).max(1), L.argmax(1)


PU = sorted(set(ports.tolist()))
held = {a: [] for a in ALPHAS}
audit = {a: [] for a in ALPHAS}
covr = {a: [] for a in ALPHAS}
for p in PU:
    src = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(src) < 3000 or len(te) < 100:
        continue
    cs, cp, ct, ci = [], [], [], []
    for q in [x for x in PU if x != p][:2]:
        trq = np.where(known & (ports != p) & (ports != q))[0]
        teq = np.where(known & (ports == q))[0]
        if len(trq) < 3000 or len(teq) < 50:
            continue
        s, pr = fit(trq, teq)
        cs.append(s); cp.append(pr); ct.append(y[teq]); ci.append(teq)
    if not cs:
        continue
    cs = np.concatenate(cs); cp = np.concatenate(cp); ct = np.concatenate(ct); cidx = np.concatenate(ci)
    cdom = LogisticRegression(max_iter=1000).fit(
        np.vstack([X[cidx], X[te]]), np.r_[np.zeros(len(cidx)), np.ones(len(te))]).predict_proba(X[cidx])[:, 1]
    cerr = (cp != ct).astype(int)
    # the error model: score + label-free surrogates -> P(error). Labels are the CALIBRATION ports' own labels, which a
    # source-side deployment has; the target port contributes neither labels nor these features.
    g = LogisticRegression(max_iter=1000).fit(np.c_[cs, cdom, (1 - cs) * cdom], cerr)
    s_t, pr_t = fit(src, te)
    dom_t = LogisticRegression(max_iter=1000).fit(
        np.vstack([X[cidx], X[te]]), np.r_[np.zeros(len(cidx)), np.ones(len(te))]).predict_proba(X[te])[:, 1]
    pred_t = g.predict_proba(np.c_[s_t, dom_t, (1 - s_t) * dom_t])[:, 1]
    ok = (pr_t == y[te])
    err_t = (~ok).astype(int)
    if 0 < err_t.sum() < len(err_t):
        a_diag = roc_auc_score(err_t, pred_t)      # source-fit error model on the TARGET port
    else:
        a_diag = float('nan')
    for a in ALPHAS:
        o = np.argsort(-s_t)
        cum = np.cumsum(pred_t[o]) / (np.arange(len(o)) + 1)
        fine = np.where(cum <= a)[0]
        acc = np.zeros(len(o), bool)
        if len(fine):
            acc[o[:fine.max() + 1]] = True
        held[a].append(bool(acc.sum() >= 20 and (~ok[acc]).mean() <= a + 1e-9))
        covr[a].append(float(acc.mean()))
        audit[a].append(a_diag)
    print('%-16s 错误模型 AUC %.3f | α=0.05 守约 %s 覆盖 %.3f' % (p, a_diag, held[0.05][-1], covr[0.05][-1]), flush=True)

print('')
print('=== 诊断：源港拟合的错误模型在目标港的 AUC ===')
au = np.array(audit[0.05]); au = au[np.isfinite(au)]
print('中位 %.3f ｜ p10 %.3f ｜ p90 %.3f ｜ >0.65 的港 %d/%d'
      % (np.median(au), np.percentile(au, 10), np.percentile(au, 90), int((au > 0.65).sum()), len(au)))
print('')
print('=== 真风险 <= α 的港数 ===')
print('%-8s %14s %10s %12s' % ('alpha', '守约港数/23', '守约率', '中位覆盖率'))
for a in ALPHAS:
    h = held[a]
    print('%-8.2f %14s %9.0f%% %12.3f' % (a, '%d/%d' % (sum(h), len(h)), 100 * sum(h) / len(h), float(np.median(covr[a]))))
print('')
print('预注册判据（真风险<=α 的港 >=80%%，α=0.05）: %.0f%% ⇒ %s'
      % (100 * sum(held[0.05]) / len(held[0.05]),
         '成立 ✓✓' if sum(held[0.05]) >= 0.8 * len(held[0.05]) else '未成立 ✗'))

"""e139: predictive phase diagram -- can the SIGN of the knowledge gain be predicted from label-free quantities?

The negative taxonomy lists what fails; a phase diagram would say when the knowledge will help. Only label-free predictors are
allowed (a deployment has no target labels), and the target -- the measured per-port gain sign -- is used for evaluation only.

Predictors (all label-free):
  prior_tv_pred   total-variation distance between the target port's PREDICTED class histogram and the source's true histogram
  disagree        fraction of target instances where the two arms disagree
  margin_spread   spread (p90-p10) of the visual margin on the target port
  src_gain        the gain the expert shows on held-out SOURCE ports (the deployable expectation)
  gate_conc       mean |gate score - 0.5| (how decisive the gate is, label-free)

Target: sign of BA(V+K) - BA(V) on that port.
Model: logistic regression fitted leave-one-port-out; reported as sign accuracy, with a binomial test against chance.
Pre-registered: accuracy >= 70% and binomial p < 0.05.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
PU = sorted(set(ports.tolist()))


def fit(trs, te, use_k):
    Z = np.c_[X[trs], K[trs]] if use_k else X[trs]
    Q = np.c_[X[te], K[te]] if use_k else X[te]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


feat, sign, names = [], [], ['prior_tv_pred', 'disagree', 'margin_spread', 'src_gain']
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    pv, mv = fit(src, te, False)
    pk, mk = fit(src, te, True)
    bV, bK = ba(y[te], pv), ba(y[te], pk)
    # label-free predictors
    h_pred = np.bincount(pk, minlength=C).astype(float); h_pred /= h_pred.sum()
    h_src = np.bincount(y[src], minlength=C).astype(float); h_src /= h_src.sum()
    tv = 0.5 * float(np.abs(h_pred - h_src).sum())
    dis = float((pv != pk).mean())
    spread = float(np.percentile(mv, 90) - np.percentile(mv, 10))
    # source-side expectation: an inner fold's gain
    q = [x for x in PU if x != p][0]
    trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
    sq = ba(y[teq], fit(trq, teq, True)[0]) - ba(y[teq], fit(trq, teq, False)[0]) if len(teq) >= 50 else 0.0
    feat.append([tv, dis, spread, sq]); sign.append(1 if bK > bV else 0)
    print('%-16s ΔBA %+.4f ｜ TV_pred %.3f 不一致 %.3f 源增益 %+.4f' % (p, bK - bV, tv, dis, sq), flush=True)

F = np.array(feat, float); s = np.array(sign)
F = (F - F.mean(0)) / (F.std(0) + 1e-9)
print('')
print('正增益港 %d/%d' % (s.sum(), len(s)))
correct = 0
for i in range(len(s)):
    m = np.ones(len(s), bool); m[i] = False
    base = 1 if s[m].mean() > 0.5 else 0
    if m.sum() >= 5:
        lr = LogisticRegression(max_iter=500, C=1.0).fit(F[m], s[m])
        pred = int(lr.predict(F[~m])[0])
    else:
        pred = base
    correct += int(pred == s[i])
print('留一符号预测正确 %d/%d = %.1f%%' % (correct, len(s), 100.0 * correct / len(s)))
bt = stats.binomtest(correct, len(s), 0.5, alternative='greater')
print('二项检验 p=%.4f' % bt.pvalue)
print('')
# which single predictor correlates with the sign?
for j, nm in enumerate(names):
    if F[:, j].std() > 0:
        r = stats.pointbiserialr(s, F[:, j])
        print('  %-14s 与符号相关 r=%+.3f p=%.4f' % (nm, r.statistic, r.pvalue))
print('')
print('预注册判据（≥70%% 且 p<0.05）: %s' % ('成立 ✓✓' if (100.0 * correct / len(s) >= 70 and bt.pvalue < 0.05) else '未成立 ✗'))

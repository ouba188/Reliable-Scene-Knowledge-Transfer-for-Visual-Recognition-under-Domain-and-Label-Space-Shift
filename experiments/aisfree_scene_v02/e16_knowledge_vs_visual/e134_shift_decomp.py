"""e134: attribute the cross-port drop -- how much is label (base-rate) shift vs conditional shift?

The law says the VALUE does not transfer. This asks WHY, by decomposing the port-to-port accuracy drop into the standard
components, using target labels for EVALUATION only (never for fitting or calibration):

  source-side accuracy  ->  accuracy with the SOURCE conditionals reweighted by the TARGET class priors  ->  actual target accuracy
        \_______ label/base-rate shift _______/                        \____ residual: conditional shift P(y|x) ____/

Pre-registered: report the shares. If label shift explains >= 50% of the drop, the follow-up prediction is that a marginal
correction should partially restore calibration (which the earlier BBSE failure contradicts unless the confusion matrix is
non-transferable); if the residual dominates, marginal-only corrections are ruled out in principle.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
PU = sorted(set(ports.tolist()))


def fit_pred(trs, te):
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mu) / sd, y[trs])
    return m.predict((X[te] - mu) / sd)


src_acc, ls_acc, tgt_acc = [], [], []
prior_tv, calib_before, calib_after = [], [], []
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 50:
        continue
    # source-side reference: one held-out source port
    q = [x for x in PU if x != p][0]
    trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
    if len(trq) < 500 or len(teq) < 50:
        continue
    yq = y[teq]; yt = y[te]
    pq = fit_pred(trq, teq); pt = fit_pred(tr, te)
    a_src = float((pq == yq).mean()); a_tgt = float((pt == yt).mean())
    # confusion on the source-side reference, row-normalised = the conditionals P(pred | true)
    R = np.zeros((C, C))
    for t in range(C):
        m = yq == t
        if m.sum():
            for k in range(C):
                R[t, k] = (pq[m] == k).mean()
    pi_s = np.array([(yq == t).mean() for t in range(C)])
    pi_t = np.array([(yt == t).mean() for t in range(C)])
    # accuracy under the target priors with the SOURCE conditionals: diagonal of R weighted by pi_t
    a_ls = float((np.diag(R) * pi_t).sum() / max(pi_t.sum(), 1e-9))
    src_acc.append(a_src); ls_acc.append(a_ls); tgt_acc.append(a_tgt)
    prior_tv.append(0.5 * float(np.abs(pi_t - pi_s).sum()))
    print('%-16s n=%5d 源参考 %.3f → 基率校正 %.3f → 目标实测 %.3f ｜ 先验 TV %.3f'
          % (p, len(te), a_src, a_ls, a_tgt, prior_tv[-1]), flush=True)

s, l, t = np.array(src_acc), np.array(ls_acc), np.array(tgt_acc)
drop = s - t
ls_part = s - l
print('')
print('源参考 %.4f → 基率校正 %.4f → 目标实测 %.4f ｜ 总掉分 %.2fpp（%d 港）'
      % (s.mean(), l.mean(), t.mean(), drop.mean() * 100, len(s)))
print('基率偏移贡献 %+.2fpp ｜ 条件偏移残差 %+.2fpp ｜ 基率占比 %.0f%%'
      % (ls_part.mean() * 100, drop.mean() * 100 - ls_part.mean() * 100,
         100 * max(0.0, ls_part.mean()) / max(drop.mean(), 1e-9)))
print('先验 TV 距离 中位 %.3f' % np.median(prior_tv))
print('')
share = 100 * max(0.0, ls_part.mean()) / max(drop.mean(), 1e-9)
print('⇒ %s' % ('基率偏移主导（>=50%%）⇒ 边际校正应有部分效果，与 BBSE 失败形成矛盾，需查混淆矩阵可迁移性 ✓'
                if share >= 50 else '条件偏移主导 ⇒ 只看边际的校正原则上无救 ✗（BBSE 类方法被原则性否掉 ✓）'))

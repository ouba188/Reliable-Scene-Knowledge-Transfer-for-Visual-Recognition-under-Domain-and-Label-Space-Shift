"""e64: is the SOURCE-side criterion predictive of the TARGET-side outcome?

Reads e63_preds_n8.pkl (per fold: the target decisions + the per-inner-source-port arms from the nested
evaluation). Question: if a rule were to pick source ports by what they look like on the source side,
would that tell us anything about the target? If not, 'autonomous source selection' must be reported as
a negative result (consistent with e38/e39), not claimed as a method feature.
"""
import pickle
from collections import Counter

import numpy as np
from scipy import stats

C = 8
d = pickle.load(open('e63_preds_n8.pkl', 'rb'))     # our own artifact, written by e63 on this machine
dump, src = d['dump'], d['src_log']
print('folds:', len(dump), '| src_log:', len(src))


def ba(yy, p):
    rs = [float((p[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


xs_worst, xs_mean, ys = [], [], []
for rec, s in zip(dump, src):
    y = rec['y']
    # target-side: the nested-free gate applied on the target = gate_new column already computed in e63;
    # recompute here from the stored scores for the two gates, plus the ridge_V reference
    ref = ba(y, rec['pred']['ridge_V'])
    gnew = ba(y, np.array([rec['pred'][c][i] for i, c in enumerate([['ridge_V', 'ridge_VK', 'gbm_zk'][j] for j in rec['sc_new'].argmax(1)])]))
    gain_t = (gnew - ref) * 100
    per = [v for v in s['per_src'].values()]
    gains_s = [(v['gate'] - v['ba_v']) * 100 for v in per]
    xs_worst.append(min(gains_s)); xs_mean.append(float(np.mean(gains_s))); ys.append(gain_t)

xs_worst, xs_mean, ys = map(np.array, (xs_worst, xs_mean, ys))
print()
print('目标港增益: mean %+.2f  sd %.2f  负港 %d/%d' % (ys.mean(), ys.std(), int((ys < 0).sum()), len(ys)))
print('源端最差内层港增益: mean %+.2f  sd %.2f' % (xs_worst.mean(), xs_worst.std()))
print('源端平均内层港增益: mean %+.2f  sd %.2f' % (xs_mean.mean(), xs_mean.std()))
print()
for nm, x in [('源端最差港准则', xs_worst), ('源端平均准则', xs_mean)]:
    pr = stats.pearsonr(x, ys); sp = stats.spearmanr(x, ys)
    print('%-14s vs 目标增益: Pearson r=%+.3f (p=%.3f)  Spearman ρ=%+.3f (p=%.3f)' % (
        nm, pr.statistic, pr.pvalue, sp.statistic, sp.pvalue))
print()
thr = np.median(xs_worst)
print('用源端最差准则做"要不要用知识"的二分（阈值=中位数 %+.2f）:' % thr)
for lab, m in [('源端判定OK', xs_worst > thr), ('源端判定差', xs_worst <= thr)]:
    print('  %-10s 目标港增益 mean %+.2f  负港 %d/%d' % (lab, ys[m].mean(), int((ys[m] < 0).sum()), int(m.sum())))

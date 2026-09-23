"""e60dbg: one-fold forensics on why the 3-way gate collapses (Antwerp-Bruges).

Prints, for the single fold: each candidate's accuracy on the target, the source-trained correctness
predictors' mean scores on the target, the pick distribution, and the resulting accuracy -- to see
whether the labels, the features or the selection are at fault.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
SUB = 20000
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def cand_preds(tr, te):
    trs = tr if len(tr) <= SUB else tr[np.isin(tr, rng.choice(tr, SUB, replace=False))]
    A, B = kstd(trs, te)
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[trs], y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[X[trs], A], y[trs])
    gk = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(np.c_[X[trs], A], y[trs])
    return {'ridge_V': (softmax(rv.decision_function(X[te])), rv.classes_),
            'ridge_VK': (softmax(rk.decision_function(np.c_[X[te], B])), rk.classes_),
            'gbm_zk': (gk.predict_proba(np.c_[X[te], B]), gk.classes_)}


def gfeats(P):
    fs = []
    for nm in CANDS:
        p = P[nm][0]; srt = np.sort(p, 1)
        fs.append((srt[:, -1] - srt[:, -2])[:, None])
        fs.append((-(p * np.log(p + 1e-12)).sum(1))[:, None])
    pa = [P[nm][0].argmax(1) for nm in CANDS]
    for i in range(len(CANDS)):
        for j in range(i + 1, len(CANDS)):
            fs.append((pa[i] == pa[j]).astype(float)[:, None])
    return np.concatenate(fs, 1)


p = 'Antwerp-Bruges'
tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
F, L, src_ba = [], [], []
for q in [x for x in PORT_U if x != p][:3]:
    trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
    Pq = cand_preds(trq, teq)
    lab = [(Pq[nm][1][Pq[nm][0].argmax(1)] == y[teq]).astype(int) for nm in CANDS]
    F.append(gfeats(Pq)); L.append(lab)
    src_ba.append([(Pq[nm][1][Pq[nm][0].argmax(1)] == y[teq]).mean() for nm in CANDS])
F = np.vstack(F)
Lc = [np.concatenate([L[i][k2] for i in range(len(L))]) for k2 in range(len(CANDS))]
print('源端各候选正确率(每个内层港):')
for i, r_ in enumerate(src_ba):
    print('   port%s %s' % (i, ' '.join('%s=%.3f' % (CANDS[j], r_[j]) for j in range(3))))
print('源端标签正例率:', ['%s=%.3f' % (CANDS[j], Lc[j].mean()) for j in range(3)])
print('源端特征形状', F.shape, '均值', np.round(F.mean(0), 3))

Pt = cand_preds(tr, te)
pa = {nm: Pt[nm][1][Pt[nm][0].argmax(1)] for nm in CANDS}
print()
print('目标港各候选 BA:', ['%s=%.3f' % (nm, ba(y[te], pa[nm])) for nm in CANDS])
models = [HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(F, lc) for lc in Lc]
Ft = gfeats(Pt)
sc = np.stack([m.predict_proba(Ft)[:, 1] for m in models], 1)
print('目标港上三模型的平均预测正确率:', np.round(sc.mean(0), 3))
print('             取 argmax 的分布:', np.round(np.bincount(sc.argmax(1), minlength=3) / len(sc), 3))
pick = sc.argmax(1)
g3 = np.array([pa[CANDS[i]][j] for j, i in enumerate(pick)])
print('gate3 BA = %.3f' % ba(y[te], g3))
print()
print('逐条对齐（前 8 条）: y | gbm_zk | ridge_V | ridge_VK | pick | gate_out')
for i in range(8):
    print('   %d | %d | %d | %d | %s | %d' % (y[te][i], pa['gbm_zk'][i], pa['ridge_V'][i], pa['ridge_VK'][i], CANDS[pick[i]], g3[i]))
print('数组类型:', type(pa['gbm_zk']), pa['gbm_zk'].dtype, '| pick dtype', pick.dtype, '| g3 len', len(g3), 'te len', len(te))
print('（对照：随机选 BA≈%.3f，全选 gbm_zk=%.3f）' % (
    np.mean([ba(y[te], pa[nm]) for nm in CANDS]), ba(y[te], pa['gbm_zk'])))

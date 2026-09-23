"""e60: 3-way gate -- make the JUDGE STRENGTH itself a per-instance choice.

Candidates per instance: ridge_V (safest tail), ridge_VK_pct (the +3.40/-5.91 config), gbm_zk
(+7.31/-19.84). The gate predicts, from source LOO episodes, which candidate is correct and picks the
argmax of those probabilities. Everything is selected on source ports only.

Reference frontier: ridge_VK +3.40/-5.91  |  gbm_zk +7.31/-19.84  |  2-way gate +8.57/-17.81
Goal: land BETWEEN them on the mean while pulling the tail back towards -6.
"""
import csv, os
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
SUBS = int(os.environ.get('E60_SUBS', '20000'))
NINNER = int(os.environ.get('E60_NINNER', '3'))
rng = np.random.default_rng(0)
CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
ARMS = CANDS + ['gate3', 'gate2']
print('cands', CANDS, 'inner', NINNER, 'subs', SUBS, flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def cand_preds(tr, te, sub=None, pca=None):
    """returns dict cand -> (pred, prob matrix)"""
    trs = tr if (sub is None or len(tr) <= sub) else tr[np.isin(tr, rng.choice(tr, sub, replace=False))]
    A, B = kstd(trs, te)
    a = 1.0                                            # alpha 固定（其余实验已确认 0.3–3 之间不敏感）
    rv = RidgeClassifier(alpha=a, class_weight='balanced').fit(X[trs], y[trs])
    rk = RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[trs], A], y[trs])
    gk = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(np.c_[X[trs], A], y[trs])
    dv = rv.decision_function(X[te]); dk = rk.decision_function(np.c_[X[te], B]); dg = gk.predict_proba(np.c_[X[te], B])
    return {'ridge_V': (softmax(dv), rv.classes_),
            'ridge_VK': (softmax(dk), rk.classes_),
            'gbm_zk': (dg, gk.classes_)}


def gfeats(P):
    """features for the correctness predictors: margins/entropies/agreements of all candidates"""
    fs = []
    for nm in CANDS:
        p = P[nm][0]
        srt = np.sort(p, 1)
        fs.append((srt[:, -1] - srt[:, -2])[:, None])                 # margin
        fs.append((-(p * np.log(p + 1e-12)).sum(1))[:, None])          # entropy
    pa = [P[nm][0].argmax(1) for nm in CANDS]
    for i in range(len(CANDS)):
        for j in range(i + 1, len(CANDS)):
            fs.append((pa[i] == pa[j]).astype(float)[:, None])
    return np.concatenate(fs, 1)


out = {a: [] for a in ARMS}
print('%-18s %7s %8s %7s %7s %7s' % tuple(['port'] + ARMS))
print('-' * 62)
for p in PORT_U:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    F, L = [], []
    for q in [x for x in PORT_U if x != p][:NINNER]:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        Pq = cand_preds(trq, teq, sub=SUBS)
        F.append(gfeats(Pq))
        L.append([(Pq[nm][1][Pq[nm][0].argmax(1)] == y[teq]).astype(int) for nm in CANDS])   # 用类别 id 而非 argmax 位置
    Pt = cand_preds(tr, te)
    Ft = gfeats(Pt)
    preds = {}
    for k2, nm in enumerate(CANDS):
        preds[nm] = Pt[nm][1][Pt[nm][0].argmax(1)]
    if F:
        F = np.vstack(F)
        Lc = [np.concatenate([L[i][k2] for i in range(len(L))]) for k2 in range(len(CANDS))]
        models = [HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(F, lc) for lc in Lc]
        scores = np.stack([m.predict_proba(Ft)[:, 1] for m in models], 1)
        pa = {nm: Pt[nm][1][Pt[nm][0].argmax(1)] for nm in CANDS}      # 各候选的预测标签
        pick3 = scores.argmax(1)
        preds['gate3'] = np.array([pa[CANDS[k3]][j] for j, k3 in enumerate(pick3)])
        pick2 = 1 + (scores[:, 2] > scores[:, 1]).astype(int)          # 只在 ridge_VK / gbm_zk 间选
        preds['gate2'] = np.array([pa[CANDS[k2]][j] for j, k2 in enumerate(pick2)])
    else:
        preds['gate3'] = preds['ridge_VK']; preds['gate2'] = preds['gbm_zk']
    for nm in ARMS:
        out[nm].append(ba(y[te], preds[nm]))
    print('%-18s %7.3f %8.3f %7.3f %7.3f %7.3f' % (p, *[out[nm][-1] for nm in ARMS]), flush=True)

print()
for nm in ARMS:
    v = np.array(out[nm]); d = (v - np.array(out['ridge_V'])) * 100
    print('%-9s mean %.4f  Δ vs ridge_V %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
print()
g3 = np.array(out['gate3']); g2 = np.array(out['gate2']); rk = np.array(out['ridge_VK']); gz = np.array(out['gbm_zk'])
print('配对 gate3 − ridge_VK: %+6.2f pp  p=%.4f' % ((g3 - rk).mean() * 100, stats.wilcoxon(g3, rk).pvalue))
print('配对 gate3 − gbm_zk  : %+6.2f pp  p=%.4f' % ((g3 - gz).mean() * 100, stats.wilcoxon(g3, gz).pvalue))
print('风险-收益前沿: ridge_VK %.4f/%+.2f | gbm_zk %.4f/%+.2f | gate2 %.4f/%+.2f | gate3 %.4f/%+.2f' % (
    rk.mean(), (rk - np.array(out['ridge_V'])).min() * 100,
    gz.mean(), (gz - np.array(out['ridge_V'])).min() * 100,
    g2.mean(), (g2 - np.array(out['ridge_V'])).min() * 100,
    g3.mean(), (g3 - np.array(out['ridge_V'])).min() * 100))

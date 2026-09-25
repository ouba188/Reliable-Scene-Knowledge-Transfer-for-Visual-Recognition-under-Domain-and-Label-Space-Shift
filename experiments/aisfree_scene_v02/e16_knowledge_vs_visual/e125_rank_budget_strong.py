"""e125: does the budget curve survive a different EXPERT PAIR? (robustness of E18p)

E18p passed with the visual/gated-knowledge ridge pair on the old pool. The new pool cannot be used yet: mapping the 82-dim
knowledge onto it is blocked by the open coordinate-frame problem from e85 (the object table's world_x/y frame and the
facility layer's lon/lat disagree by 50-90 km), so "re-verify on the new pool" is not a 30-minute job.

The cheap, unblocked robustness test: keep the pool and the protocol, swap the expert. e59's strong judge (a GBM on
visual + knowledge) has a much larger gain (+8.88 pp against the ridge's +3.32 pp), so the budget-value curve is measured with
a different expert and a different effect size. Same pre-registered criterion: some k in {5,10,20}% with rank beating random
at paired p<0.05, and rank@20% recovering >= 50% of the full-expert gain.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz')['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
BUDGETS = (0.05, 0.10, 0.20, 0.50, 1.00)
NINNER, NREP = 3, 20
rng = np.random.default_rng(0)
print('池 %d | %d 港 | 专家=GBM(视觉+知识) [e59 强判据]' % (len(y), len(set(ports.tolist()))), flush=True)
Zk = np.c_[X, K]


def arm_models(trs):
    mz = X[trs].mean(0, keepdims=True); sz = X[trs].std(0, keepdims=True) + 1e-6
    mv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mz) / sz, y[trs])
    mzk = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(Zk[trs], y[trs])
    return (mv, mz, sz), mzk


def preds(models, te):
    (mv, mz, sz), mzk = models
    Lv = mv.decision_function((X[te] - mz) / sz)
    Sv = softmax(Lv, 1); t2 = np.sort(Lv, 1)[:, -2:]
    Lk = mzk.predict_proba(Zk[te]); ck = mzk.classes_
    Pk = np.zeros((len(te), C)); Pk[:, ck] = Lk
    return (Lv.argmax(1), t2[:, 1] - t2[:, 0], -(Sv * np.log(Sv + 1e-12)).sum(1)), Pk.argmax(1), Pk.max(1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
per_k = {b: {'rank': [], 'rand': []} for b in BUDGETS}
fullgain = []
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 20:
        continue
    F, L = [], []
    for q in [x for x in PU if x != p][:NINNER]:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 500 or len(teq) < 20:
            continue
        v, pk, ck = preds(arm_models(trq), teq)
        F.append(np.c_[v[1], v[2], ck, (v[0] == pk).astype(float)])
        L.append(((pk == y[teq]) & (v[0] != y[teq])).astype(int))
    if not F:
        continue
    F = np.vstack(F); L = np.concatenate(L)
    v, pk, ck = preds(arm_models(tr), te)
    yy = y[te]; bV = ba(yy, v[0])
    fullgain.append(ba(yy, pk) - bV)
    Ft = np.c_[v[1], v[2], ck, (v[0] == pk).astype(float)]
    g = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=0).fit(F, L)
    s = g.predict_proba(Ft)[:, 1]
    for b in BUDGETS:
        n = max(1, int(b * len(te)))
        if b >= 1.0:
            per_k[b]['rank'].append(ba(yy, pk) - bV); per_k[b]['rand'].append(ba(yy, pk) - bV)
            continue
        sel = np.zeros(len(te), bool); sel[np.argsort(-s)[:n]] = True
        per_k[b]['rank'].append(ba(yy, np.where(sel, pk, v[0])) - bV)
        rr = [ba(yy, np.where(np.isin(np.arange(len(te)), rng.choice(len(te), n, replace=False)), pk, v[0])) - bV
              for _ in range(NREP)]
        per_k[b]['rand'].append(float(np.mean(rr)))
    print('%-16s n=%5d 全视觉 %.3f' % (p, len(te), bV), flush=True)

print('')
print('%-8s %12s %12s %12s %10s' % ('预算 k', '秩选择 ΔBA', '随机 ΔBA', '差值', '配对 p'))
for b in BUDGETS:
    a = np.array(per_k[b]['rank']); r = np.array(per_k[b]['rand'])
    pv = stats.wilcoxon(a, r).pvalue if len(a) >= 8 and np.any(a != r) else float('nan')
    print('%-8s %+12.4f %+12.4f %+12.4f %10.4f' % ('%.0f%%' % (100 * b), a.mean(), r.mean(), (a - r).mean(), pv))
fg = float(np.mean(fullgain))
print('')
print('全专家增益（GBM 强判据）: %+.4f' % fg)
for b in (0.05, 0.10, 0.20, 0.50):
    a = np.array(per_k[b]['rank'])
    print('  k=%2.0f%% 秩选择 %+.4f ⇒ 吃下 %.1f%%' % (100 * b, a.mean(), 100 * a.mean() / fg))
ok = any(np.array(per_k[b]['rank']).mean() > np.array(per_k[b]['rand']).mean()
         and stats.wilcoxon(per_k[b]['rank'], per_k[b]['rand']).pvalue < 0.05 for b in (0.05, 0.10, 0.20))
rec = 100 * np.array(per_k[0.20]['rank']).mean() / fg
print('')
print('预注册判据（某个 k 秩>随机 p<0.05 且 rank@20%% 收复 >=50%%）: %s（判据1 %s ｜ 收复 %.1f%%）'
      % ('成立 ✓✓' if (ok and rec >= 50) else '未成立 ✗', '✓' if ok else '✗', rec))

"""e127: does the budget curve hold across DIFFERENT EXPERT DEFINITIONS?

The new-pool re-verification is not available: the first 42 knowledge dims are per-facility-type proximities plus local
background contrasts, which need OSM facility layers for every port, and only 8 ports have them locally. So the independent
replication uses what is already here -- the knowledge blocks from e15 (facility 0:42, chain 42:60, detection 60:71, all legal
0:71; the 71:82 dims stay out to keep the protocol clean, per e69's legal range).

Each block becomes the expert: an expert is the visual+block classifier whose consultation the budget policy allocates. Same
pre-registered criterion per expert: some k in {5,10,20}% with rank beating random at paired p<0.05, and rank@20% recovering
>= 50% of that expert's full gain.
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
Kraw = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
BLOCKS = {'facility': (0, 42), 'chain': (42, 60), 'detection': (60, 71), 'all_legal': (0, 71)}
BUDGETS = (0.05, 0.10, 0.20, 0.50, 1.00)
NINNER, NREP = 3, 20
rng = np.random.default_rng(0)


def fit_arm(trs, te, K):
    Z = np.c_[X[trs], K[trs]]
    Q = np.c_[X[te], K[te]]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0], -(S * np.log(S + 1e-12)).sum(1)


def fit_v(trs, te):
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mu) / sd, y[trs])
    L = m.decision_function((X[te] - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0], -(S * np.log(S + 1e-12)).sum(1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
summary = {}
for bn, (a0, a1) in BLOCKS.items():
    K = Kraw[:, a0:a1]
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
            pv, mv, ev = fit_v(trq, teq)
            pk, mk, ek = fit_arm(trq, teq, K)
            F.append(np.c_[mv, mk, ek, ev, (pv == pk).astype(float)])
            L.append(((pk == y[teq]) & (pv != y[teq])).astype(int))
        if not F:
            continue
        F = np.vstack(F); L = np.concatenate(L)
        pv, mv, ev = fit_v(tr, te)
        pk, mk, ek = fit_arm(tr, te, K)
        Ft = np.c_[mv, mk, ek, ev, (pv == pk).astype(float)]
        yy = y[te]; bV = ba(yy, pv)
        fullgain.append(ba(yy, pk) - bV)
        g = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=0).fit(F, L)
        s = g.predict_proba(Ft)[:, 1]
        for b in BUDGETS:
            n = max(1, int(b * len(te)))
            if b >= 1.0:
                per_k[b]['rank'].append(ba(yy, pk) - bV); per_k[b]['rand'].append(ba(yy, pk) - bV)
                continue
            sel = np.zeros(len(te), bool); sel[np.argsort(-s)[:n]] = True
            per_k[b]['rank'].append(ba(yy, np.where(sel, pk, pv)) - bV)
            per_k[b]['rand'].append(float(np.mean([
                ba(yy, np.where(np.isin(np.arange(len(te)), rng.choice(len(te), n, replace=False)), pk, pv)) - bV
                for _ in range(NREP)])))
    fg = float(np.mean(fullgain))
    summary[bn] = {'fg': fg, 'k': {}}
    for b in BUDGETS:
        a = np.array(per_k[b]['rank']); r = np.array(per_k[b]['rand'])
        pv2 = stats.wilcoxon(a, r).pvalue if len(a) >= 8 and np.any(a != r) else float('nan')
        summary[bn]['k'][b] = (a.mean(), r.mean(), pv2, a.mean() / fg if fg else float('nan'))
    ok = any(summary[bn]['k'][b][0] > summary[bn]['k'][b][1] and summary[bn]['k'][b][2] < 0.05 for b in (0.05, 0.10, 0.20))
    rec = 100 * summary[bn]['k'][0.20][3]
    print('%-10s 全专家增益 %+.4f | 判据 %s ｜ rank@20%% 收复 %.1f%%'
          % (bn, fg, '✓' if (ok and rec >= 50) else '✗', rec), flush=True)

print('')
print('=== 各专家块 × 预算 ===')
print('%-10s %9s %6s %12s %12s %10s %10s' % ('专家块', '全专家增益', '预算', '秩 ΔBA', '随机 ΔBA', '收复', '配对 p'))
for bn in BLOCKS:
    for b in BUDGETS:
        a, r, pv2, rec = summary[bn]['k'][b]
        print('%-10s %+9.4f %5.0f%% %+12.4f %+12.4f %9.1f%% %10.4f'
              % (bn, summary[bn]['fg'], 100 * b, a, r, 100 * rec, pv2))
print('')
n_ok = 0
for bn in BLOCKS:
    ok = any(summary[bn]['k'][b][0] > summary[bn]['k'][b][1] and summary[bn]['k'][b][2] < 0.05 for b in (0.05, 0.10, 0.20))
    rec = 100 * summary[bn]['k'][0.20][3]
    n_ok += bool(ok and rec >= 50)
print('判据通过的知识块: %d/%d ｜ %s' % (n_ok, len(BLOCKS),
                                       '模块对专家定义不敏感 ✓✓' if n_ok >= 3 else '对专家定义敏感 ⚠'))

"""e124: rank-based BUDGETED allocation -- using the one thing that does transfer (the order).

Tonight's law (E18o): cross-port transfer preserves the order and destroys the value, so every threshold/guarantee
construction failed while every ranking measurement succeeded. That suggests the move nobody tried: never transfer a
threshold at all. Transfer the ORDER, and make the BUDGET the policy knob --

    consulted = the top-k% of the target port by gate score      (pure rank, no calibration, no target labels)

Testable, and none of the earlier gates tested it (they all used absolute thresholds). Pre-registered criterion:
  for at least one k in {5,10,20}%, the mean gain of top-k% selection must beat RANDOM k% selection with paired p<0.05
  over the 24 ports, AND rank@20% must recover >= 50% of the full-knowledge gain.
If the rank carries decision value, top-k% selection beats random-k% -- that is what 'the order transfers' has to mean
operationally. Pool: the 38,091-chip 8-class pool, the only one carrying the 82-dim knowledge.
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
print('池 %d | %d 港 | %d 类 | 知识 %d 维' % (len(y), len(set(ports.tolist())), C, K.shape[1]), flush=True)


def std_fit(trs, te, use_k):
    Z = np.c_[X[trs], K[trs]] if use_k else X[trs]
    Q = np.c_[X[te], K[te]] if use_k else X[te]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1)
    top2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), (top2[:, 1] - top2[:, 0]), (-(S * np.log(S + 1e-12)).sum(1))


def gate_feats(pv, mv, ev, pk, mk, ek, kk):
    return np.c_[mv, mk, ek, ev, (pv == pk).astype(float), (mk - mv), np.abs(mk - mv),
                 np.linalg.norm(kk, axis=1), (np.linalg.norm(kk, axis=1) > 0).astype(float)]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
res = {k2: [] for k2 in ('rank', 'rand', 'allV', 'allVK')}
per_k = {b: {'rank': [], 'rand': []} for b in BUDGETS}
for p in PU:
    tr = np.where(ports != p)[0]
    te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 20:
        continue
    F, L = [], []
    for q in [x for x in PU if x != p][:NINNER]:
        trq = np.where((ports != p) & (ports != q))[0]
        teq = np.where(ports == q)[0]
        if len(trq) < 500 or len(teq) < 20:
            continue
        pv, mv, ev = std_fit(trq, teq, False)
        pk, mk, ek = std_fit(trq, teq, True)
        F.append(gate_feats(pv, mv, ev, pk, mk, ek, K[teq]))
        L.append(((pk == y[teq]) & (pv != y[teq])).astype(int))     # 'consulting was the right call'
    if not F:
        continue
    F = np.vstack(F); L = np.concatenate(L)
    pv, mv, ev = std_fit(tr, te, False)
    pk, mk, ek = std_fit(tr, te, True)
    Ft = gate_feats(pv, mv, ev, pk, mk, ek, K[te])
    yy = y[te]
    bV, bK = ba(yy, pv), ba(yy, pk)
    res['allV'].append(bV); res['allVK'].append(bK)
    for b in BUDGETS:
        n = max(1, int(b * len(te)))
        if b >= 1.0:
            pr = pk
        else:
            g = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=0).fit(F, L)
            s = g.predict_proba(Ft)[:, 1]
            sel = np.zeros(len(te), bool); sel[np.argsort(-s)[:n]] = True
            pr = np.where(sel, pk, pv)
        per_k[b]['rank'].append(ba(yy, pr) - bV)
        if b >= 1.0:
            per_k[b]['rand'].append(ba(yy, pk) - bV)
        else:
            rr = [ba(yy, np.where(np.isin(np.arange(len(te)), rng.choice(len(te), n, replace=False)), pk, pv)) - bV
                  for _ in range(NREP)]
            per_k[b]['rand'].append(float(np.mean(rr)))
    print('%-16s n=%5d | 全视觉 %.3f 全知识 %.3f' % (p, len(te), bV, bK), flush=True)

print('')
print('%-8s %12s %12s %12s %10s' % ('预算 k', '秩选择 ΔBA', '随机 ΔBA', '差值', '配对 p'))
for b in BUDGETS:
    a = np.array(per_k[b]['rank']); r = np.array(per_k[b]['rand'])
    pv = stats.wilcoxon(a, r).pvalue if len(a) >= 8 else float('nan')
    print('%-8s %+12.4f %+12.4f %+12.4f %10.4f' % ('%.0f%%' % (100 * b), a.mean(), r.mean(), (a - r).mean(), pv))
full = np.array(res['allVK']) - np.array(res['allV'])
print('')
print('全知识增益（参照）: %+.4f' % full.mean())
for b in (0.05, 0.10, 0.20):
    a = np.array(per_k[b]['rank'])
    print('  k=%2.0f%% 秩选择 %.4f ⇒ 吃下全知识增益的 %.1f%%' % (100 * b, a.mean(), 100 * a.mean() / full.mean()))
ok = any(np.array(per_k[b]['rank']).mean() > np.array(per_k[b]['rand']).mean()
         and stats.wilcoxon(per_k[b]['rank'], per_k[b]['rand']).pvalue < 0.05 for b in (0.05, 0.10, 0.20))
rec = 100 * np.array(per_k[0.20]['rank']).mean() / full.mean()
print('')
print('预注册判据（某个 k 秩>随机 p<0.05 且 rank@20%% 收复 >=50%%）: %s（判据1 %s ｜ 收复 %.1f%%）'
      % ('成立 ✓✓' if (ok and rec >= 50) else '未成立 ✗', '✓' if ok else '✗', rec))

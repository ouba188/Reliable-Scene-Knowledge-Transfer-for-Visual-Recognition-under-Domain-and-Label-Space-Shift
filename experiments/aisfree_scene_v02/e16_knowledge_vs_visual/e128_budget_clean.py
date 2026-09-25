"""e128: the compliant version of the budget module, done properly -- effect sizes with intervals, more power.

E18s withdrew E18p/E18q's significance: their experts included the AIS-derived dims 71:82, which the P0 AIS-free protocol
forbids. This runs the clean expert (knowledge dims 0:71 only) with the power fixes:

  - inner calibration episodes 3 -> 6, random-size control redraws 20 -> 50 (a steadier random baseline means the paired
    comparison is less noisy, which is where the missing power was);
  - effect sizes with 24-port bootstrap 95% intervals reported FIRST, p-values second;
  - two clean experts: ridge(visual+legal) and GBM(visual+legal).

Pre-registered: at a 20% budget the rank-minus-random effect must have a bootstrap 95% CI excluding 0, and rank@20% must
recover >= 50% of the clean expert's full gain.
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
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)[:, 0:71]     # LEGAL: P0 AIS-free
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
BUDGETS = (0.05, 0.10, 0.20, 0.50, 1.00)
NINNER, NREP = 6, 50
rng = np.random.default_rng(0)
print('合规知识 %d 维（P0 AIS-free ✓）| 内层 %d ｜ 重抽 %d' % (K.shape[1], NINNER, NREP), flush=True)


def fit_v(trs, te):
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mu) / sd, y[trs])
    L = m.decision_function((X[te] - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0], -(S * np.log(S + 1e-12)).sum(1)


def fit_k(trs, te, gbm):
    Z = np.c_[X[trs], K[trs]]; Q = np.c_[X[te], K[te]]
    if gbm:
        m = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(Z, y[trs])
        P = m.predict_proba(Q); cl = m.classes_
        Pf = np.zeros((len(te), C)); Pf[:, cl] = P
        return Pf.argmax(1), Pf.max(1), -(Pf * np.log(Pf + 1e-12)).sum(1)
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0], -(S * np.log(S + 1e-12)).sum(1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
for expert, gbm in (('ridge_legal', False), ('gbm_legal', True)):
    per_k = {b: {'rank': [], 'rand': []} for b in BUDGETS}
    fg = []
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
            pk, mk, ek = fit_k(trq, teq, gbm)
            F.append(np.c_[mv, mk, ek, ev, (pv == pk).astype(float)])
            L.append(((pk == y[teq]) & (pv != y[teq])).astype(int))
        if not F:
            continue
        F = np.vstack(F); L = np.concatenate(L)
        pv, mv, ev = fit_v(tr, te)
        pk, mk, ek = fit_k(tr, te, gbm)
        Ft = np.c_[mv, mk, ek, ev, (pv == pk).astype(float)]
        yy = y[te]; bV = ba(yy, pv)
        fg.append(ba(yy, pk) - bV)
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
        print('  %-11s %-16s n=%5d' % (expert, p, len(te)), flush=True)
    fgm = float(np.mean(fg))
    print('')
    print('=== 专家 %s ｜ 全专家增益 %+.4f（合规 ✓）===' % (expert, fgm))
    print('%-6s %18s %10s %12s %10s' % ('预算', '秩−随机 (95% CI)', 'p', '收复', 'CI 排除 0'))
    for b in BUDGETS:
        a = np.array(per_k[b]['rank']); r = np.array(per_k[b]['rand'])
        d = a - r
        if b >= 1.0:
            print('%-6s %18s' % ('100%', '—'))
            continue
        bs = np.array([np.mean(rng.choice(d, len(d))) for _ in range(5000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pv2 = stats.wilcoxon(a, r).pvalue if np.any(d != 0) else float('nan')
        rec = 100 * a.mean() / fgm if fgm else float('nan')
        print('%-6s %+8.4f [%+.4f,%+.4f] %10.4f %11.1f%% %10s'
              % ('%.0f%%' % (100 * b), d.mean(), lo, hi, pv2, rec, '✓' if lo > 0 else '✗'))
    a20 = np.array(per_k[0.20]['rank']); r20 = np.array(per_k[0.20]['rand'])
    d20 = a20 - r20
    bs = np.array([np.mean(rng.choice(d20, len(d20))) for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rec20 = 100 * a20.mean() / fgm
    print('  20%% 判据（CI 排除 0 且 收复 >=50%%）: CI [%+.4f,%+.4f] ｜ 收复 %.1f%% ⇒ %s'
          % (lo, hi, rec20, '成立 ✓✓' if (lo > 0 and rec20 >= 50) else '未成立 ✗'))
    print('', flush=True)

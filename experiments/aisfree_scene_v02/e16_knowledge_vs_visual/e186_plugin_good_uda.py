"""e186: the decisive add-on test -- does the policy help a UDA method that is ALREADY positive?

E24b showed the policy repairs a harmful adaptation (CORAL, net negative here). That is damage control, not improvement of a good
method, and the difference decides whether the add-on is a methods contribution. So repeat with the most standard UDA baseline
that can actually work in this setting: pseudo-labelling / self-training, which needs no gradient access and is label-free at
decision time.

Arms per port (old pool, leave-one-port-out):
  V        ridge on the source only
  PL       self-training: take the target's most confident alpha-quantile under V, add them with pseudo-labels, refit, predict
  PL+bud   our policy on top of PL: rank the target by PL's own margin, use PL's prediction for the top k%, V otherwise
  random   matched-size random mixture, the floor

Pre-registered: (i) report whether PL itself is positive against V; (ii) at k = 20%, PL+bud must beat PL-everywhere in >= 2/3
ports AND pooled AND beat the random floor.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
ART = BASE / 'artifacts/eight_class_adaptive_20260916'
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
KS = (0.10, 0.20, 0.50)
ALPHAS = (0.30, 0.50)
NREP = 20
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


def fit(Xtr, ytr):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Xtr - mu) / sd, ytr)
    return m, mu, sd


PU = sorted(set(ports.tolist()))
res = {a: {k: [] for k in KS} for a in ALPHAS}
pl_gain = {a: [] for a in ALPHAS}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 50:
        continue
    m, mu, sd = fit(X[tr], y[tr])
    Zt = (X[te] - mu) / sd
    predV = m.predict(Zt)
    L = m.decision_function(Zt)
    t2 = np.sort(L, 1)[:, -2:]
    marginV = t2[:, 1] - t2[:, 0]
    yy = y[te]
    for a in ALPHAS:
        nsel = max(1, int(a * len(te)))
        idx = np.argsort(-marginV)[:nsel]
        Xtr2 = np.vstack([(X[tr] - mu) / sd, Zt[idx]])
        ytr2 = np.concatenate([y[tr], predV[idx]])
        m2, mu2, sd2 = fit(Xtr2, ytr2)
        Zt2 = (X[te] - mu2) / sd2
        predPL = m2.predict(Zt2)
        L2 = m2.decision_function(Zt2)
        t22 = np.sort(L2, 1)[:, -2:]
        marginPL = t22[:, 1] - t22[:, 0]
        pl_gain[a].append(ba(yy, predPL) - ba(yy, predV))
        for k in KS:
            n = max(1, int(k * len(te)))
            sel = np.zeros(len(te), bool); sel[np.argsort(-marginPL)[:n]] = True
            res[a][k].append(dict(
                V=ba(yy, predV), PL=ba(yy, predPL),
                bud=ba(yy, np.where(sel, predPL, predV)),
                rand=float(np.mean([(lambda msk: ba(yy, np.where(msk, predPL, predV)))(
                    np.isin(np.arange(len(te)), rng.permutation(len(te))[:n])) for _ in range(NREP)]))))
    print('%-16s n=%5d ｜ V %.3f' % (p, len(te), ba(yy, predV)), flush=True)

for a in ALPHAS:
    g = np.array(pl_gain[a])
    print('')
    print('=== 伪标 α=%.2f ｜ PL 相对 V 的 Δ 中位 %+.4f ｜ 正港 %d/%d ===' % (a, np.median(g), int((g > 0).sum()), len(g)))
    print('%-8s %10s %10s %12s %10s %10s' % ('预算', 'V', 'PL', 'PL+预算', '随机', 'p(预算−PL)'))
    for k in KS:
        d = res[a][k]
        V = np.mean([x['V'] for x in d]); PL = np.mean([x['PL'] for x in d])
        B = np.mean([x['bud'] for x in d]); R = np.mean([x['rand'] for x in d])
        bud = np.array([x['bud'] for x in d]); pl = np.array([x['PL'] for x in d])
        pv = stats.wilcoxon(bud, pl).pvalue if np.any(bud != pl) else float('nan')
        print('%-8s %10.4f %10.4f %12.4f %10.4f %10.4f' % ('%.0f%%' % (100 * k), V, PL, B, R, pv))
    d = res[a][0.20]
    bud = np.array([x['bud'] for x in d]); pl = np.array([x['PL'] for x in d])
    rnd = np.array([x['rand'] for x in d])
    wins = int((bud > pl).sum()); pv = stats.wilcoxon(bud, pl).pvalue
    ok = wins >= max(2, int(np.ceil(len(bud) * 2 / 3))) and pv < 0.05 and bud.mean() > rnd.mean()
    print('20%%：插件更好 %d/%d 港 ｜ Δ %+.4f ｜ p=%.4f ｜ 优于随机地板 %s ⇒ 判据 %s'
          % (wins, len(bud), (bud - pl).mean(), pv, '✓' if bud.mean() > rnd.mean() else '✗',
             '成立 ✓✓' if ok else '未成立 ✗'))

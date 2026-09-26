"""e185: is the policy an ADD-ON to domain adaptation? Test it on top of CORAL.

Standard UDA baselines change the model or the features. Our policy changes neither, so the claim to test is that it adds value
downstream of an adaptation method -- here plain CORAL (whiten the target features, recolour them with the source covariance),
implemented in a few lines so no dependency is needed.

Arms, leave-one-port-out on the old pool:
  V            ridge on source, target features untouched
  V+CORAL      the same ridge, but the target features CORAL-aligned (the UDA baseline, applied everywhere)
  V+CORAL+bud  our policy on top: rank the target port by the CORAL arm's own margin and use the CORAL prediction for the top
               k%, the plain prediction otherwise (so the "expert" being consulted is the adapted arm)
  random       matched-size random subset of the same mixture, as the floor

Pre-registered: at k = 20%, the add-on must beat CORAL-everywhere in >= 2 of 3 folds AND pooled (paired over 24 ports); the
random floor must be below the add-on.
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
NREP = 20
rng = np.random.default_rng(0)


def coral(Xs, Xt):
    """align the target covariance to the source's: whiten the target, recolour with the source."""
    ms = Xs.mean(0, keepdims=True); mt = Xt.mean(0, keepdims=True)
    Cs = np.cov((Xs - ms).T) + 1e-3 * np.eye(Xs.shape[1])
    Ct = np.cov((Xt - mt).T) + 1e-3 * np.eye(Xs.shape[1])
    def sqrtm_inv(A):
        w, V = np.linalg.eigh(A)
        return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-9))) @ V.T, V @ np.diag(np.sqrt(np.maximum(w, 1e-9))) @ V.T
    Ct_inv, Cs_sqrt = sqrtm_inv(Ct)[0], sqrtm_inv(Cs)[1]
    return (Xt - mt) @ Ct_inv @ Cs_sqrt + ms


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
res = {k: {a: [] for a in ('plain', 'coral', 'bud', 'rand')} for k in KS}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 50:
        continue
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[tr] - mu) / sd, y[tr])
    Zt = (X[te] - mu) / sd
    pred_plain = m.predict(Zt)
    Xt_c = coral((X[tr] - mu) / sd, Zt)
    pred_coral = m.predict(Xt_c)
    Lc = m.decision_function(Xt_c)
    t2 = np.sort(Lc, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    yy = y[te]
    for k in KS:
        n = max(1, int(k * len(te)))
        sel = np.zeros(len(te), bool); sel[np.argsort(-margin)[:n]] = True
        mix = np.where(sel, pred_coral, pred_plain)
        res[k]['plain'].append(ba(yy, pred_plain))
        res[k]['coral'].append(ba(yy, pred_coral))
        res[k]['bud'].append(ba(yy, mix))
        # the random floor mixes the SAME two predictions on a random subset of the same size
        res[k]['rand'].append(float(np.mean([
            (lambda m: ba(yy, np.where(m, pred_coral, pred_plain)))(
                np.isin(np.arange(len(te)), rng.permutation(len(te))[:n])) for _ in range(NREP)])))
    print('%-16s n=%5d ｜ V %.3f ｜ CORAL %.3f' % (p, len(te), ba(yy, pred_plain), ba(yy, pred_coral)), flush=True)

print('')
print('%-8s %10s %10s %10s %10s %10s' % ('预算', 'V', 'CORAL', 'CORAL+预算', '随机', 'p(预算−CORAL)'))
for k in KS:
    a = {x: np.array(res[k][x]) for x in res[k]}
    pv = stats.wilcoxon(a['bud'], a['coral']).pvalue if np.any(a['bud'] != a['coral']) else float('nan')
    print('%-8s %10.4f %10.4f %10.4f %10.4f %10.4f' % ('%.0f%%' % (100 * k), a['plain'].mean(), a['coral'].mean(),
                                                       a['bud'].mean(), a['rand'].mean(), pv))
a = {x: np.array(res[0.20][x]) for x in res[0.20]}
wins = int((a['bud'] > a['coral']).sum())
pv = stats.wilcoxon(a['bud'], a['coral']).pvalue
print('')
print('20%% 处：加插件后更好 %d/%d 港 ｜ Δ %+.4f ｜ p=%.4f ｜ 插件是否优于随机地板 %s'
      % (wins, len(a['bud']), (a['bud'] - a['coral']).mean(), pv, '✓' if a['bud'].mean() > a['rand'].mean() else '✗'))
ok = wins >= max(2, int(np.ceil(len(a['bud']) * 2 / 3))) and pv < 0.05 and a['bud'].mean() > a['rand'].mean()
print('预注册判据（20%% 处 ≥2/3 港占优 且 p<0.05 且 优于随机地板）: %s'
      % ('成立 ✓✓ —— 它是 UDA 的可叠加下游插件 ✓' if ok else '未成立 ✗ —— 不能宣称可叠加 ✓'))

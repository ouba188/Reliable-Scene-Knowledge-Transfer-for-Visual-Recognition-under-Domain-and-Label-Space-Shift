"""e119: class-conditional conformal rejection -- the first candidate whose metric is well powered.

Metric: risk-coverage frontier of selective classification over 55,102 known + 16,699 unknown instances -- not the 1.2% rare
event that killed every earlier module. It rides the one well-powered positive signal found so far: the class-conditional
Mahalanobis novelty score.

Test-time reality: the class is unknown, only predicted, so the per-class threshold must be keyed on the PREDICTION; that key
noise is the interesting part, and the oracle-per-true-class arm bounds how much of it is recoverable.

Arms: global (one threshold) / predclass (threshold of the PREDICTED class) / trueclass (oracle key, upper bound).
Metric: AURC = mean risk over coverage in [0,1], lower is better.
Pre-registered: AURC(predclass) < AURC(global), paired p < 0.05 over 24 ports, closing >= 50% of the global -> trueclass gap.

ponytail: features are projected to 512 dims once (e75 measured pca512 ~= raw: 0.4722 vs 0.4735), and the source calibration
uses 2 held-out ports instead of 6 -- both to fit inside the 0.2-1.2 GB of free RAM this box has; the ceiling of the
simplification is a slightly noisier calibration threshold, which the trueclass arm bounds anyway.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
CACHE = ROOT / 'features_all' / 'pca512_all.npy'
if CACHE.exists():
    X = np.load(CACHE).astype(np.float32)
else:
    Xf = np.load(ROOT / 'features_all' / 'resnet50_c64.float16.npy').astype(np.float32)
    X = PCA(n_components=512, svd_solver='randomized', random_state=0).fit_transform(Xf).astype(np.float32)
    np.save(CACHE, X)
    del Xf
print('特征 %s | %d 维' % (X.shape, X.shape[1]), flush=True)
rows = list(csv.DictReader((ROOT / 'dataset_all' / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
y = np.full(len(rows), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
known = y >= 0
ALPHA = 0.05
N_CAL_PORT = 2         # inner calibration ports (LOO-clean: never the target)


def fit_score(trs, te):
    """class-conditional Mahalanobis novelty for te (higher = more conforming), fitted on trs only."""
    Z = X[trs]
    mu = Z.mean(0, keepdims=True); Zc = Z - mu
    C = Zc.T @ Zc / len(Z)
    C = 0.95 * C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C.astype(np.float64)).astype(np.float32)
    means = np.stack([Z[y[trs] == j].mean(0) for j in range(8)])
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[trs])
    Q = X[te]
    d = np.stack([np.einsum('ij,jk,ik->i', Q - means[j], P, Q - means[j]) for j in range(8)])
    return -d.min(0), clf.predict(Q)


def calib_threshold(s, err, alpha):
    """conformal risk control: the largest accepted set whose empirical risk stays <= alpha."""
    o = np.argsort(-s); s, err = s[o], err[o]
    cum = np.cumsum(err) / (np.arange(len(err)) + 1)
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


def aurc(score, correct):
    o = np.argsort(-score)
    return float((np.cumsum(1 - correct[o]) / (np.arange(len(correct)) + 1)).mean())


ARMS = ('global', 'predclass', 'trueclass')
arms = {a: [] for a in ARMS}
PU = sorted(set(ports.tolist()))
for p in PU:
    src = np.where(known & (ports != p))[0]
    te = np.where(known & (ports == p))[0]
    if len(src) < 3000 or len(te) < 100:
        continue
    sc, sp, st = [], [], []
    for q in [x for x in PU if x != p][:N_CAL_PORT]:
        trq = np.where(known & (ports != p) & (ports != q))[0]
        teq = np.where(known & (ports == q))[0]
        if len(trq) < 3000 or len(teq) < 50:
            continue
        s, pr = fit_score(trq, teq)
        sc.append(s); sp.append(pr); st.append(y[teq])
    if not sc:
        continue
    sc = np.concatenate(sc); sp = np.concatenate(sp); st = np.concatenate(st)
    err = (sp != st).astype(float)
    t_glob = calib_threshold(sc, err, ALPHA)
    t_pre = {j: (calib_threshold(sc[sp == j], err[sp == j], ALPHA) if (sp == j).sum() >= 100 else t_glob)
             for j in range(8)}
    t_tru = {j: (calib_threshold(sc[st == j], err[st == j], ALPHA) if (st == j).sum() >= 100 else t_glob)
             for j in range(8)}
    s, pr = fit_score(src, te)
    yy = y[te]; acc = (pr == yy).astype(float)
    arms['global'].append(aurc(s, acc))
    arms['predclass'].append(aurc(np.array([s[i] - t_pre[pr[i]] for i in range(len(te))]), acc))
    arms['trueclass'].append(aurc(np.array([s[i] - t_tru[yy[i]] for i in range(len(te))]), acc))
    print('%-16s n=%5d 准确率 %.3f | AURC 全局 %.4f 逐预测类 %.4f 逐真类 %.4f'
          % (p, len(te), acc.mean(), arms['global'][-1], arms['predclass'][-1], arms['trueclass'][-1]), flush=True)

print('')
g = np.array(arms['global']); pc = np.array(arms['predclass']); tc = np.array(arms['trueclass'])
for nm, v in (('全局', g), ('逐预测类', pc), ('逐真类(上界)', tc)):
    print('%-12s AURC %.4f   Δ vs 全局 %+.5f' % (nm, v.mean(), (v - g).mean()))
p1 = stats.wilcoxon(pc, g).pvalue
gap = (g - tc).mean()
close = (g - pc).mean() / gap if gap > 0 else float('nan')
print('')
print('配对 逐预测类 − 全局: %+.5f  p=%.4f' % ((pc - g).mean(), p1))
print('收复比例（全局→真类 差距吃下多少）: %.1f%%' % (100 * close))
print('预注册判据（逐预测类 < 全局 且 p<0.05 且 收复 >=50%%）: %s'
      % ('成立 ✓✓' if (pc.mean() < g.mean() and p1 < 0.05 and close >= 0.5) else '未成立 ✗'))

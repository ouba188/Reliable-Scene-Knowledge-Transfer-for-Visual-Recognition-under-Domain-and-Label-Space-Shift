"""e121: C3R paper-Table-1 -- all four hardening checks in one pass.

The module (per-predicted-class conformal rejection) already passes two pre-registered criteria. Hardening, to the standard a
reviewer would demand:
  (1) alternative ACCEPTANCE SCORES -- the module must add value on top of more than one score, otherwise it is just a
      by-product of the Mahalanobis construction. Scores: maha / max-softmax / neg-energy / ridge margin.
  (2) ALPHA SWEEP (0.01/0.05/0.10) and the conformal GUARANTEE check: is the realised risk among accepted instances at the
      target port actually <= alpha?
  (3) CALIBRATION-PORT robustness (N_CAL from the env, 2 vs 4).
  (4) per-port x per-class risk grid at a fixed coverage (by-product of the same run).

Naming: key modes are global (one threshold) and predclass (threshold of the predicted class -- the module).
Evaluation is the open-set reading (unknown instances count as errors) over all instances of the target port.
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from scipy import stats
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
X = np.load(ROOT / 'features_all' / 'pca512_all.npy').astype(np.float32)
rows = list(csv.DictReader((ROOT / 'dataset_all' / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
y = np.full(len(rows), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
known = y >= 0
N_CAL = int(os.environ.get('N_CAL', '2'))
ALPHAS = (0.01, 0.05, 0.10)
SCORES = ('maha', 'softmax', 'negenergy', 'margin')
COV_GRID = 0.9


def fit(trs, te):
    Z = X[trs]
    mu = Z.mean(0, keepdims=True); Zc = Z - mu
    C = Zc.T @ Zc / len(Z)
    C = 0.95 * C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C.astype(np.float64)).astype(np.float32)
    means = np.stack([Z[y[trs] == j].mean(0) for j in range(8)])
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[trs])
    Q = X[te]
    d = np.stack([np.einsum('ij,jk,ik->i', Q - means[j], P, Q - means[j]) for j in range(8)])
    L = clf.decision_function(Q)
    S = softmax(L, 1)
    top2 = np.sort(L, 1)[:, -2:]
    return {'maha': -d.min(0), 'softmax': S.max(1), 'negenergy': -logsumexp(L, 1),
            'margin': top2[:, 1] - top2[:, 0]}, clf.classes_[L.argmax(1)]


def thr(s, err, alpha):
    o = np.argsort(-s); s, err = s[o], err[o]
    cum = np.cumsum(err) / (np.arange(len(err)) + 1)
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


def aurc(s, ok):
    o = np.argsort(-s)
    return float((np.cumsum(1 - ok[o]) / (np.arange(len(ok)) + 1)).mean())


PU = sorted(set(ports.tolist()))
res = {(sc, km, a): [] for sc in SCORES for km in ('global', 'predclass') for a in ALPHAS}
guar = {(sc, km, a): [] for sc in SCORES for km in ('global', 'predclass') for a in ALPHAS}
grid = []
for p in PU:
    src = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(src) < 3000 or len(te) < 100:
        continue
    cal = {sc: [] for sc in SCORES}; cpred = []; ctru = []
    for q in [x for x in PU if x != p][:N_CAL]:
        trq = np.where(known & (ports != p) & (ports != q))[0]
        teq = np.where(known & (ports == q))[0]
        if len(trq) < 3000 or len(teq) < 50:
            continue
        s, pr = fit(trq, teq)
        for sc in SCORES:
            cal[sc].append(s[sc])
        cpred.append(pr); ctru.append(y[teq])
    if not cpred:
        continue
    cpred = np.concatenate(cpred); ctru = np.concatenate(ctru)
    cerr = (cpred != ctru).astype(float)
    st, pr = fit(src, te)
    yy = y[te]; ok = (pr == yy)
    for sc in SCORES:
        s = st[sc]; cs = np.concatenate(cal[sc])
        for a in ALPHAS:
            tg = thr(cs, cerr, a)
            tp = {j: (thr(cs[cpred == j], cerr[cpred == j], a) if (cpred == j).sum() >= 100 else tg) for j in range(8)}
            for km, v in (('global', s - tg), ('predclass', np.array([s[i] - tp[pr[i]] for i in range(len(te))]))):
                res[(sc, km, a)].append(aurc(v, ok))
                acc = v >= 0
                if acc.sum() >= 20:
                    guar[(sc, km, a)].append(float((~ok[acc]).mean()))
                if a == 0.05:
                    o = np.argsort(-v); k = max(1, int(COV_GRID * len(o)))
                    sel = o[:k]
                    for c in range(8):
                        m = sel[yy[sel] == c]
                        if len(m) >= 20:
                            grid.append((p, KNOWN8[c], float((~ok[m]).mean()), len(m)))
    print('%-16s done (%d/%d)' % (p, len(res[("maha", "global", 0.05)]), len(PU)), flush=True)

print('')
print('=== (1)+(2) 分数 × 键 × α：AURC 与实现风险 ===')
print('%-10s %-10s %5s %10s %10s %10s %10s' % ('score', 'key', 'alpha', 'AURC', 'Δ vs global', '实现风险', '守约'))
base = {(sc, a): np.array(res[(sc, 'global', a)]) for sc in SCORES for a in ALPHAS}
for sc in SCORES:
    for km in ('global', 'predclass'):
        for a in ALPHAS:
            v = np.array(res[(sc, km, a)]); b = base[(sc, a)]
            gr = np.mean(guar[(sc, km, a)]) if guar[(sc, km, a)] else float('nan')
            okg = '✓' if gr <= a + 1e-9 else '✗'
            print('%-10s %-10s %5.2f %10.4f %+10.4f %10.4f %10s'
                  % (sc, km, a, v.mean(), (v - b).mean(), gr, okg))
print('')
print('=== 配对检验（逐预测类 − 全局，α=0.05）===')
for sc in SCORES:
    g = np.array(res[(sc, 'global', 0.05)]); pc = np.array(res[(sc, 'predclass', 0.05)])
    print('%-10s Δ=%+.5f  p=%.4f  %s' % (sc, (pc - g).mean(), stats.wilcoxon(pc, g).pvalue,
                                         '成立 ✓' if (pc.mean() < g.mean() and stats.wilcoxon(pc, g).pvalue < 0.05) else '不成立 ✗'))
print('')
print('=== (4) 逐港×逐类风险网格（90%% 覆盖，逐预测类，α=0.05）：%d 格 ===' % len(grid))
arr = np.array([[g[2]] for g in grid]).ravel()
print('风险 中位 %.3f ｜ p10 %.3f ｜ p90 %.3f ｜ 最差港类 %s %s (%.3f, n=%d)'
      % (np.median(arr), np.percentile(arr, 10), np.percentile(arr, 90),
         *sorted(grid, key=lambda g: -g[2])[0][:2],
         sorted(grid, key=lambda g: -g[2])[0][2], sorted(grid, key=lambda g: -g[2])[0][3]))
per_cls = {}
for _, c, r, n in grid:
    per_cls.setdefault(c, []).append(r)
print('%-26s %6s %10s' % ('已知类', '港数', '中位风险'))
for c in KNOWN8:
    if c in per_cls:
        print('%-26s %6d %10.3f' % (c, len(per_cls[c]), float(np.median(per_cls[c]))))

"""e120: the open-set risk curve -- unknown instances counted as errors, over all 71,801 instances.

e119 passed its criterion on known-class instances only (standard selective classification). But 23% of this pool is
out-of-distribution by label space, and a closed-set classifier MUST mislabel those instances -- it has no label for them.
Counting them as errors is the open-set reading, and it turns e119's two separate measurements (novelty AUC 0.555 for
unknown rejection, AURC 0.5876 for known selective risk) into one curve over the whole pool, which is also better powered.

Thresholds stay calibrated on SOURCE known-class instances (the deployment situation: source labels only), applied unchanged
to a target port that mixes 8 known and 9 unknown classes.

Arms: global / predclass (keyed on the PREDICTION) / trueclass (oracle key).
Pre-registered: AURC(predclass) < AURC(global) with paired p < 0.05 over the 24 ports, AND predclass beats global at the
90%-coverage operating point (where unknown rejection does the work).
"""
import csv
from pathlib import Path

import numpy as np
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
ALPHA, N_CAL_PORT, COV = 0.05, 2, (0.5, 0.7, 0.9)


def fit_score(trs, te):
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
    o = np.argsort(-s); s, err = s[o], err[o]
    cum = np.cumsum(err) / (np.arange(len(err)) + 1)
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


def aurc(score, correct):
    o = np.argsort(-score)
    return float((np.cumsum(1 - correct[o]) / (np.arange(len(correct)) + 1)).mean())


def risk_at(score, correct, cov):
    o = np.argsort(-score)
    k = max(1, int(cov * len(o)))
    return float((1 - correct[o][:k]).mean())


arms = {a: [] for a in ('global', 'predclass', 'trueclass')}
covr = {a: {c: [] for c in COV} for a in arms}
PU = sorted(set(ports.tolist()))
for p in PU:
    src = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]                      # ALL instances: known and unknown
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
    yy = y[te]
    ok = (pr == yy)                     # unknown instances (y<0) are automatically wrong: a closed-set classifier must mislabel
    sc_dict = {'global': s,
               'predclass': np.array([s[i] - t_pre[pr[i]] for i in range(len(te))]),
               'trueclass': np.array([s[i] - t_tru[yy[i]] if yy[i] >= 0 else s[i] - t_glob for i in range(len(te))])}
    for a, v in sc_dict.items():
        arms[a].append(aurc(v, ok))
        for c in COV:
            covr[a][c].append(risk_at(v, ok, c))
    print('%-16s n=%5d 未知占比 %.2f | AURC 全局 %.4f 逐预测类 %.4f 逐真类 %.4f'
          % (p, len(te), 1 - known[te].mean(), arms['global'][-1], arms['predclass'][-1], arms['trueclass'][-1]), flush=True)

print('')
g = np.array(arms['global']); pc = np.array(arms['predclass']); tc = np.array(arms['trueclass'])
for nm, v in (('全局', g), ('逐预测类', pc), ('逐真类(上界)', tc)):
    print('%-12s AURC %.4f   Δ vs 全局 %+.5f' % (nm, v.mean(), (v - g).mean()))
print('')
print('%-10s %10s %10s %10s %10s' % ('覆盖率', '全局风险', '逐预测类', '逐真类', 'Δ(逐预测类−全局)'))
for c in COV:
    print('%-10s %10.4f %10.4f %10.4f %+10.4f' % ('%.0f%%' % (100 * c),
          np.mean(covr['global'][c]), np.mean(covr['predclass'][c]), np.mean(covr['trueclass'][c]),
          np.mean(np.array(covr['predclass'][c]) - np.array(covr['global'][c]))))
p1 = stats.wilcoxon(pc, g).pvalue
p2 = stats.wilcoxon(covr['predclass'][0.9], covr['global'][0.9]).pvalue
print('')
print('配对 AURC 逐预测类 − 全局: %+.5f  p=%.4f' % ((pc - g).mean(), p1))
print('配对 90%%覆盖风险 逐预测类 − 全局: %+.5f  p=%.4f'
      % ((np.array(covr['predclass'][0.9]) - np.array(covr['global'][0.9])).mean(), p2))
print('预注册判据（AURC 更低 p<0.05 且 90%%覆盖点更优 p<0.05）: %s'
      % ('成立 ✓✓' if (pc.mean() < g.mean() and p1 < 0.05 and p2 < 0.05) else '未成立 ✗'))

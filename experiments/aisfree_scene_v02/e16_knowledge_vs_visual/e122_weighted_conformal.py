"""e122: shift-aware weighted conformal -- fixing the measured failure.

e121 measured the disease precisely: realised risk among accepted instances is 0.12-0.63 against targets alpha=0.01/0.05/0.10,
because cross-port transfer breaks exchangeability, so the source-calibrated threshold does not transfer. The standard repair is
to weight the calibration samples by the estimated target/source density ratio before running conformal risk control.

Weights are estimated without any target label: a domain classifier (source calibration features vs target-port features)
gives p(target | x), and w = p / (1 - p) is the density ratio, clipped for stability. The target port contributes NO labels --
only its unlabelled features, which a deployment has.

Pre-registered criterion: realised risk <= alpha on >= 80% of the 24 ports (the unweighted construction scores 0%).
Also reported: the coverage price (weighted calibration rejects more in absolute terms) and the paired risk change.
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeClassifier

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
W_CLIP = (0.05, 20.0)


def fit(trs, te):
    Z = X[trs]
    mu = Z.mean(0, keepdims=True); C = ((Z - mu).T @ (Z - mu)) / len(Z)
    C = 0.95 * C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    P = np.linalg.inv(C.astype(np.float64)).astype(np.float32)
    means = np.stack([Z[y[trs] == j].mean(0) for j in range(8)])
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Z, y[trs])
    Q = X[te]
    d = np.stack([np.einsum('ij,jk,ik->i', Q - means[j], P, Q - means[j]) for j in range(8)])
    L = clf.decision_function(Q)
    return {'maha': -d.min(0), 'softmax': softmax(L, 1).max(1)}, clf.classes_[L.argmax(1)]


def thr_unw(s, err, alpha):
    o = np.argsort(-s); s, err = s[o], err[o]
    cum = np.cumsum(err) / (np.arange(len(err)) + 1)
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


def thr_w(s, err, w, alpha):
    o = np.argsort(-s); s, err, w = s[o], err[o], w[o]
    cw = np.cumsum(w)
    cum = np.cumsum(w * err) / cw
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


PU = sorted(set(ports.tolist()))
res = {(sc, km, a): [] for sc in ('maha', 'softmax') for km in ('unw', 'wtd') for a in ALPHAS}
cov = {(sc, km, a): [] for sc in ('maha', 'softmax') for km in ('unw', 'wtd') for a in ALPHAS}
for p in PU:
    src = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(src) < 3000 or len(te) < 100:
        continue
    cal, cpred, ctru, calidx = [], [], [], []
    for q in [x for x in PU if x != p][:N_CAL]:
        trq = np.where(known & (ports != p) & (ports != q))[0]
        teq = np.where(known & (ports == q))[0]
        if len(trq) < 3000 or len(teq) < 50:
            continue
        s, pr = fit(trq, teq)
        cal.append(s); cpred.append(pr); ctru.append(y[teq]); calidx.append(teq)
    if not cal:
        continue
    cpred = np.concatenate(cpred); ctru = np.concatenate(ctru); calidx = np.concatenate(calidx)
    cerr = (cpred != ctru).astype(float)
    Zc = np.vstack([X[calidx], X[te]])
    lab = np.r_[np.zeros(len(calidx)), np.ones(len(te))]
    lg = LogisticRegression(max_iter=300, C=1.0).fit(Zc, lab)
    ptar = lg.predict_proba(X[calidx])[:, 1]
    w = np.clip(ptar / np.maximum(1 - ptar, 1e-6), *W_CLIP)
    st, pr = fit(src, te)
    yy = y[te]; ok = (pr == yy)
    for sc in ('maha', 'softmax'):
        s = st[sc]; cs = cal[0][sc] if len(cal) == 1 else np.concatenate([c[sc] for c in cal])
        for a in ALPHAS:
            for km, t in (('unw', thr_unw(cs, cerr, a)), ('wtd', thr_w(cs, cerr, w, a))):
                v = s - t
                acc = v >= 0
                res[(sc, km, a)].append(float((~ok[acc]).mean()) if acc.sum() >= 20 else float('nan'))
                cov[(sc, km, a)].append(float(acc.mean()))
    print('%-16s done (%d)' % (p, len(res[('maha', 'unw', 0.05)])), flush=True)

print('')
print('=== 实现风险 vs 目标 α（NaN = 接纳集 <20）===')
print('%-9s %-5s %6s %12s %12s %10s %10s' % ('score', 'mode', 'alpha', '中位实现风险', '守约港数/24', '中位覆盖率', '判决'))
for sc in ('maha', 'softmax'):
    for a in ALPHAS:
        for km in ('unw', 'wtd'):
            v = np.array(res[(sc, km, a)])
            mm = np.isfinite(v)
            held = int((v[mm] <= a + 1e-9).sum())
            print('%-9s %-5s %6.2f %12.4f %12s %10.3f %10s'
                  % (sc, km, a, np.nanmedian(v) if mm.any() else float('nan'),
                     '%d/%d' % (held, len(v)), np.mean(cov[(sc, km, a)]),
                     '✓' if held >= 0.8 * len(v) else '✗'))
print('')
print('=== 预注册判据（实现风险 <= α 的港 >= 80%，α=0.05）===')
for sc in ('maha', 'softmax'):
    v = np.array(res[(sc, 'wtd', 0.05)]); mm = np.isfinite(v)
    held = int((v[mm] <= 0.05 + 1e-9).sum())
    print('%-9s 加权 %d/%d (%.0f%%) ⇒ %s' % (sc, held, len(v), 100 * held / len(v),
                                             '成立 ✓✓' if held >= 0.8 * len(v) and len(v) >= 10 else '未成立 ✗'))
print('')
print('=== 加权 vs 未加权（α=0.05，有限对）===')
for sc in ('maha', 'softmax'):
    u = np.array(res[(sc, 'unw', 0.05)]); wt = np.array(res[(sc, 'wtd', 0.05)])
    m = np.isfinite(u) & np.isfinite(wt)
    if m.sum() >= 5:
        print('%-9s 未加权中位 %.4f → 加权中位 %.4f ｜ 加权更优 %d/%d 港 ｜ Wilcoxon p=%.4f'
              % (sc, np.median(u[m]), np.median(wt[m]), int((wt[m] < u[m]).sum()), int(m.sum()),
                 stats.wilcoxon(wt[m], u[m]).pvalue))
    cu = np.array(cov[(sc, 'unw', 0.05)]); cw = np.array(cov[(sc, 'wtd', 0.05)])
    print('%-9s 覆盖率 未加权 %.3f → 加权 %.3f（代价 ✓）' % (sc, cu.mean(), cw.mean()))

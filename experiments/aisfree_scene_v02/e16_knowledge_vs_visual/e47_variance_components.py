"""e47 (c, corrected): variance components of the per-port knowledge gain, corrected for sampling noise.

The raw group-level deltas are differences of BAs computed on small groups, so their scatter mixes
real variation with binomial noise. Here we report, for each grouping level:
  observed variance  vs  expected sampling variance  ->  corrected (real) component.
Also saves per-chip flags so this is re-analysable without re-running the LOO.
"""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
prods = np.array([r['product'] for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
PORT_U = sorted(set(ports.tolist()))
FL = Path(r'E:/临时会话/visual_reliable_baseline/e47_flags.npz')


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def fit(tr, te, a):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    Ka, Kb = (A - mu) / sd, (B - mu) / sd
    pv = RidgeClassifier(alpha=a, class_weight='balanced').fit(X[tr], y[tr]).predict(X[te])
    pk = RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[tr], Ka], y[tr]).predict(np.c_[X[te], Kb])
    return pv, pk


def tune_alpha(src):
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in src[:3]:
            tr = np.where(np.isin(ports, src) & (ports != q))[0]; te = np.where(ports == q)[0]
            if len(tr) < 200 or len(set(y[tr].tolist())) < C:
                continue
            sc.append(ba(y[te], fit(tr, te, a)[1]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


if FL.exists():
    d = np.load(FL)   # npz holds bool arrays only
    V_ok, K_ok = d['v_ok'], d['k_ok']
else:
    V_ok = np.zeros(len(rows), bool); K_ok = np.zeros(len(rows), bool)
    for p in PORT_U:
        te = np.where(ports == p)[0]
        if len(te) < 10:
            continue
        a = tune_alpha([q for q in PORT_U if q != p])
        pv, pk = fit(np.where(ports != p)[0], te, a)
        V_ok[te] = pv == y[te]; K_ok[te] = pk == y[te]
    np.savez(FL, v_ok=V_ok, k_ok=K_ok)

bal = lambda m: float(K_ok[m].mean()) - float(V_ok[m].mean())     # micro delta (prob difference)
pp = lambda m: ((float(K_ok[m].mean()) - float(V_ok[m].mean())) * 100)


def se_delta(m):
    """SE of the difference of two correlated binary rates (paired)."""
    n = int(m.sum())
    if n < 5:
        return np.nan
    a, b = V_ok[m].astype(float), K_ok[m].astype(float)
    d = b - a
    return float(d.std(ddof=1) / np.sqrt(n)) * 100


print('%-18s %6s %8s %8s' % ('port', 'n', 'Δ(pp)', 'SE'))
pm = {}
for p in PORT_U:
    m = ports == p
    if m.sum() < 10:
        continue
    pm[p] = (int(m.sum()), pp(m), se_delta(m))
    print('%-18s %6d %+8.2f %8.2f' % (p, *pm[p]))

dps = np.array([v[1] for v in pm.values()]); ses = np.array([v[2] for v in pm.values()])
obs_sd = float(dps.std(ddof=1))
noise_sd = float(np.sqrt(np.mean(ses ** 2)))
real_sd = float(np.sqrt(max(0.0, obs_sd ** 2 - noise_sd ** 2)))
print()
print('港口级：观测 sd %.2f pp  |  抽样噪声 sd %.2f pp  |  校正后真实 sd %.2f pp' % (obs_sd, noise_sd, real_sd))
print('   ⇒ 港口间真实方差占总观测方差的 %.0f%%' % (100 * real_sd ** 2 / obs_sd ** 2))

g = defaultdict(list)
for p in PORT_U:
    m = ports == p
    if m.sum() < 10:
        continue
    for pr in set(prods[m]):
        mm = m & (prods == pr)
        if mm.sum() >= 5:
            g[p].append((pp(mm), se_delta(mm), int(mm.sum())))
allv = np.array([t[0] for v in g.values() for t in v]); allse = np.array([t[1] for v in g.values() for t in v])
npro = len(allv)
print()
print('产品级：观测 sd %.2f pp  |  抽样噪声 sd %.2f pp  (n=%d 产品)' % (allv.std(ddof=1), np.sqrt(np.nanmean(allse ** 2)), npro))
print('   ⇒ 产品级散点有 %.0f%% 的方差可由抽样噪声解释' % (100 * np.nanmean(allse ** 2) / allv.var(ddof=1)))

# 组内（去除港口均值后）的真实成分
within_obs, within_noise = [], []
for p, v in g.items():
    if len(v) < 3:
        continue
    a = np.array([t[0] for t in v]); s = np.array([t[1] for t in v])
    within_obs.append(a.var(ddof=1)); within_noise.append(np.nanmean(s ** 2))
wo, wn = float(np.mean(within_obs)), float(np.mean(within_noise))
print()
print('港内（产品间）：观测方差 %.1f  噪声方差 %.1f  ⇒ 真实港内方差 %.1f (占观测 %.0f%%)'
      % (wo, wn, max(0.0, wo - wn), 100 * max(0.0, wo - wn) / max(1e-9, wo)))
print()
print('结论：港口间真实 sd %.2f pp；港内真实 sd %.2f pp'
      % (real_sd, np.sqrt(max(0.0, wo - wn))))

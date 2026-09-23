"""e39: regional conditioning -- is the rule port-specific or merely region-specific?

For each target port p, three source sets, each with its own V and V+K (so Δ is comparable even
though absolute BA differs):
  global   : all other ports            (the standard protocol, reference +2.24)
  regional : ports in p's region only    (if >=2)
  randmatch: a size-matched random subset of the other ports   <- controls for "fewer sources"

If Δ_regional > Δ_global and Δ_randmatch ~ Δ_global, the rule is region-specific => knowledge
transfers within a region. If Δ_regional <= Δ_randmatch, the specificity is finer than a region.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0, 10.0]
PORT_U = sorted(set(ports.tolist()))

REGION = {
    'Qingdao': 'E-Asia', 'Ningbo-Zhoushan': 'E-Asia', 'Shanghai': 'E-Asia', 'Busan': 'E-Asia',
    'Port Klang': 'SE-Asia', 'Singapore': 'SE-Asia', 'Port Hedland': 'Oceania',
    'Sydney Botany': 'Oceania', 'Melbourne': 'Oceania', 'Newcastle': 'Oceania',
    'Jebel Ali': 'MidEast', 'Fujairah': 'MidEast', 'Tanger Med': 'Med', 'Port Said': 'Med',
    'Mombasa': 'Africa', 'Richards Bay': 'Africa', 'Callao': 'S-America', 'Santos': 'S-America',
    'Los Angeles': 'N-America', 'New York': 'N-America', 'Houston': 'N-America',
    'Rotterdam': 'Europe', 'Antwerp-Bruges': 'Europe', 'Hamburg': 'Europe',
}


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te, dims, mode, Kd):
    if not dims:
        return X[tr], X[te]
    A, B = Kd[tr][:, dims], Kd[te][:, dims]
    if mode == 'std':
        mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
        A, B = (A - mu) / sd, (B - mu) / sd
    return np.c_[X[tr], A], np.c_[X[te], B]


def eval_arm(src_ports, te, dims):
    tr = np.where(np.isin(ports, src_ports))[0]
    if len(tr) < 200 or len(set(y[tr].tolist())) < C:
        return None
    a = 1.0
    best = -1
    for al in ALPHAS:
        sc = []
        for q in [q for q in PORT_U if q in src_ports][:3]:
            itr = np.where(np.isin(ports, src_ports) & (ports != q))[0]
            ite = np.where(ports == q)[0]
            if len(itr) < 200 or len(set(y[itr].tolist())) < C:
                continue
            f1, f2 = feats(itr, ite, dims, 'std', K)
            sc.append(ba(y[ite], RidgeClassifier(alpha=al, class_weight='balanced').fit(f1, y[itr]).predict(f2)))
        if sc and float(np.mean(sc)) > best:
            best, a = float(np.mean(sc)), al
    fv1, fv2 = X[tr], X[te]
    fk1, fk2 = feats(tr, te, dims, 'std', K)
    v = ba(y[te], RidgeClassifier(alpha=a, class_weight='balanced').fit(fv1, y[tr]).predict(fv2))
    kk = ba(y[te], RidgeClassifier(alpha=a, class_weight='balanced').fit(fk1, y[tr]).predict(fk2))
    return v, kk


rng = np.random.default_rng(3)
print('%-22s %6s %8s %8s %8s %8s %8s' % ('port', 'reg', 'V_glob', 'K_glob', 'V_reg', 'K_reg', 'Δ_randcon'))
print('-' * 76)
rowsout = []
for p in PORT_U:
    te = np.where(ports == p)[0]
    others = [q for q in PORT_U if q != p]
    g = eval_arm(others, te, LEGAL)
    if g is None:
        continue
    reg = REGION.get(p, '?')
    sibs = [q for q in others if REGION.get(q) == reg]
    r = eval_arm(sibs, te, LEGAL) if len(sibs) >= 2 else None
    d = None
    if sibs:
        rs = list(rng.choice([q for q in others if q not in sibs], size=min(len(sibs), len(others) - len(sibs)), replace=False))
        d = eval_arm(rs, te, LEGAL)
    rowsout.append((p, reg, g[0], g[1], r[0] if r else float('nan'), r[1] if r else float('nan'),
                    (d[1] - d[0]) * 100 if d else float('nan')))
    print('%-22s %6s %8.3f %8.3f %8.3f %8.3f %8.2f' % (
        p, reg, g[0], g[1], r[0] if r else float('nan'), r[1] if r else float('nan'),
        (d[1] - d[0]) * 100 if d else float('nan')))

dg = np.array([(r[3] - r[2]) * 100 for r in rowsout])
dr = np.array([(r[5] - r[4]) * 100 for r in rowsout])
dd = np.array([r[6] for r in rowsout])
print()
print('Δ(K−V) 全局源港   : %+.2f pp  (%d/%d 正)' % (dg.mean(), (dg > 0).sum(), len(dg)))
print('Δ(K−V) 同区域源港 : %+.2f pp  (%d/%d 正)' % (np.nanmean(dr), (dr > 0).sum(), len(dr)))
print('Δ(K−V) 同规模随机 : %+.2f pp  (%d/%d 正)' % (np.nanmean(dd), (dd > 0).sum(), len(dd)))
print()
n = np.isfinite(dr)
print('配对比较（%d 个港）: 区域 vs 全局 均值差 %+.2f pp，区域更好的港数 %d/%d'
      % (n.sum(), (dr[n] - dg[n]).mean(), int((dr[n] > dg[n]).sum()), int(n.sum())))
print('              区域 vs 随机 均值差 %+.2f pp，区域更好的港数 %d/%d'
      % ((dr[n] - dd[n]).mean(), int((dr[n] > dd[n]).sum()), int(n.sum())))

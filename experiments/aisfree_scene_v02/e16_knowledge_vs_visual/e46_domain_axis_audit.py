"""e46 (c): domain-axis audit -- is "cross-port" actually cross-acquisition?

Question: the per-port knowledge gain delta_p varies a lot (14/24 positive). How much of that
variance is PORT versus ACQUISITION (product/year/orbit/overpass time, which are nearly constant
within a port)? If the within-port (between-product) variance is comparable, then "port" is a proxy
for the acquisition configuration and every port-level statement is confounded.

Method: standard per-port LOO to get per-chip V and V+K correctness -> group by port and by product
-> delta per group -> one-way variance decomposition (ICC) + correlation with the metadata parsed
from the Sentinel-1 product names.
"""
import csv, re, datetime as dt
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

# 城市经纬度（用于把 UTC 换算为地方太阳时，进而判升/降轨）
LON = {'Qingdao': 120.3, 'Ningbo-Zhoushan': 121.9, 'Shanghai': 121.5, 'Busan': 129.0,
       'Port Klang': 101.4, 'Singapore': 103.8, 'Port Hedland': 118.6, 'Sydney Botany': 151.2,
       'Melbourne': 144.9, 'Newcastle': 151.8, 'Jebel Ali': 55.1, 'Fujairah': 56.3,
       'Tanger Med': -5.5, 'Port Said': 32.3, 'Mombasa': 39.7, 'Richards Bay': 32.0,
       'Callao': -77.1, 'Santos': -46.3, 'Los Angeles': -118.2, 'New York': -74.0,
       'Houston': -95.0, 'Rotterdam': 4.5, 'Antwerp-Bruges': 4.4, 'Hamburg': 10.0}


def parse(p):
    m = re.match(r'(S1[AB])_(\w+)_(\w+)_(\w+)_(\d{8})T(\d{6})_\d{8}T\d{6}_(\d{6})_', p)
    if not m:
        return None
    plat, mode, ptype, pol, d, t, orb = m.groups()
    return dict(plat=plat, mode=mode, pol=pol, date=d,
                hh=int(t[:2]), mm=int(t[2:4]), orbit=int(orb))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fit(tr, te, a):
    Ka, Kb = prep(tr, te)
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


V_ok = np.zeros(len(rows), bool); K_ok = np.zeros(len(rows), bool)
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 10:
        continue
    a = tune_alpha(src)
    tr = np.where(ports != p)[0]
    pv, pk = fit(tr, te, a)
    V_ok[te] = pv == y[te]; K_ok[te] = pk == y[te]

print('=== 每港采集概要 ===')
meta = {}
for p in PORT_U:
    m = ports == p
    ms = [parse(x) for x in prods[m]]
    ms = [x for x in ms if x]
    if not ms:
        continue
    hrs = sorted({(x['hh'] + x['mm'] / 60 + LON.get(p, 0) / 15) % 24 for x in ms})
    dirs = ['desc' if 15 <= h <= 21 else ('asc' if 2 <= h <= 9 else '?') for h in hrs]
    meta[p] = dict(n=len(ms), nprod=len(set(prods[m])), plat=sorted({x['plat'] for x in ms}),
                   yrs=sorted({x['date'][:4] for x in ms}), h=float(np.mean(hrs)),
                   orb=(min(x['orbit'] for x in ms), max(x['orbit'] for x in ms)),
                   dir=sorted(set(dirs)))
    print('  %-18s 景 %3d 产品 %3d 平台 %-12s 年 %-14s 地方时 %5.2f 轨道 %6d-%6d 方向 %s' % (
        p, meta[p]['n'], meta[p]['nprod'], ','.join(meta[p]['plat']), ','.join(meta[p]['yrs']),
        meta[p]['h'], meta[p]['orb'][0], meta[p]['orb'][1], ','.join(meta[p]['dir'])))

d_port, d_prod = {}, {}
for p in PORT_U:
    m = ports == p
    if m.sum() < 10:
        continue
    d_port[p] = (float(K_ok[m].mean()) - float(V_ok[m].mean())) * 100
    for pr in set(prods[m]):
        mm = m & (prods == pr)
        if mm.sum() >= 15:
            d_prod[(p, pr)] = (float(K_ok[mm].mean()) - float(V_ok[mm].mean())) * 100

print()
print('=== 逐港 Δ（BA 差，pp）===')
for p in sorted(d_port, key=lambda q: -d_port[q]):
    print('  %-18s %+7.2f   (景 %3d, 地方时 %5.2f, 年 %s)' % (
        p, d_port[p], meta.get(p, {}).get('n', 0), meta.get(p, {}).get('h', 0), ','.join(meta.get(p, {}).get('yrs', []))))

bp = np.array(list(d_port.values()))
allp = np.array(list(d_prod.values()))
keys = list(d_prod.keys())
print()
print('港口级 Δ: 均值 %+.2f  标准差 %.2f  (n=%d)' % (bp.mean(), bp.std(), len(bp)))
print('产品级 Δ: 均值 %+.2f  标准差 %.2f  (n=%d, 每港产品数中位 %d)'
      % (allp.mean(), allp.std(), len(allp), int(np.median([meta[p]['nprod'] for p in d_port]))))

# 组内（同港不同产品）vs 组间 方差分解
gm = allp.mean()
ss_between = ss_within = 0.0
for p in d_port:
    v = np.array([d_prod[k2] for k2 in keys if k2[0] == p])
    if len(v) < 2:
        continue
    ss_between += len(v) * (v.mean() - gm) ** 2
    ss_within += ((v - v.mean()) ** 2).sum()
tot = ss_between + ss_within
print()
print('产品级 Δ 的方差分解: 港口间 SS %.0f (%.1f%%) / 港内(产品间) SS %.0f (%.1f%%)'
      % (ss_between, 100 * ss_between / tot, ss_within, 100 * ss_within / tot))
print('  ⇒ 港口解释了 %.0f%% 的 Δ 方差，其余 %.0f%% 是港内采集差异' % (100 * ss_between / tot, 100 * ss_within / tot))

# Δ 与采集轴的相关
print()
for lab, f in [('年份', lambda p: float(np.mean([int(x['date'][:4]) for x in [parse(q) for q in prods[ports == p]] if x]))),
               ('地方时', lambda p: meta.get(p, {}).get('h', np.nan)),
               ('平台S1B占比', lambda p: float(np.mean([x['plat'] == 'S1B' for x in [parse(q) for q in prods[ports == p]] if x]))),
               ('轨道中值', lambda p: float(np.median([x['orbit'] for x in [parse(q) for q in prods[ports == p]] if x])))]:
    xs = np.array([f(p) for p in d_port]); ys = np.array([d_port[p] for p in d_port])
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() > 4 and np.std(xs[ok]) > 1e-9:
        r = np.corrcoef(xs[ok], ys[ok])[0, 1]
        print('  corr(Δ_p, %-10s) = %+.3f   (p≈%.3f)' % (lab, r, 2 * (1 - abs(r)) if abs(r) < 1 else 1.0))
    else:
        print('  corr(Δ_p, %-10s) : 无变化（该轴在港间恒定）' % lab)

neg = [p for p in d_port if d_port[p] < 0]
print()
print('负增益港 %d/%d :' % (len(neg), len(d_port)), ', '.join(neg))
print('它们的采集特征:', '; '.join('%s(%s,%s,%s)' % (
    p, ','.join(meta.get(p, {}).get('plat', [])), ','.join(meta.get(p, {}).get('yrs', [])),
    ','.join(meta.get(p, {}).get('dir', []))) for p in neg))

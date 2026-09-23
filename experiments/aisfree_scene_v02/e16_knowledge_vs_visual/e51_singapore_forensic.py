"""e51: forensic on Singapore's harm -- what exactly breaks, and is it rescuable?

1. per-class V / K_pct recalls and the deltas
2. the flip anatomy: rescue vs harm counts, and WHICH confused pairs carry the harm
3. does the gate see it: per-instance AUC inside Singapore + the mean predicted rescue-prob of the
   rescued vs the harmed instances
4. support check: per-dim KS distance between Singapore's percentile knowledge and the source pool,
   compared against every other port (is Singapore out-of-support?)
5. per-product concentration of the harm
"""
import csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from scipy.stats import ks_2samp

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
feat = __import__('json').load((ROOT / 'knowledge' / 'relation_features.json').open(encoding='utf-8'))
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
prods = np.array([r['product'] for r in rows])
cname = {}
for r in rows:
    cname[int(r['class_id'])] = r['class_name']
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]; TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PORT_U = sorted(set(ports.tolist())); NSRC = 8
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
TARGET = 'Singapore'
print('=== 0) 支持度检查：各港百分位知识 vs 源港池的 KS 距离（越大越"超出源支持"）===')
ks_all = {}
for p in PORT_U:
    m = ports == p; o = ports != p
    d = np.mean([ks_2samp(Kp[m, i], Kp[o, i]).statistic for i in LEGAL])
    ks_all[p] = d
for p, d in sorted(ks_all.items(), key=lambda t: -t[1])[:6]:
    print('   %-18s KS %.4f %s' % (p, d, '  <== 目标' if p == TARGET else ''))
print('   中位 KS %.4f ⇒ Singapore 排名 %d/%d' % (np.median(list(ks_all.values())),
      sorted(ks_all.values(), reverse=True).index(ks_all[TARGET]) + 1, len(ks_all)))


def ba_mask(ok, yy):
    per = [float(ok[yy == c].mean()) for c in range(C) if (yy == c).sum() > 0]
    return float(np.mean(per)) if per else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def fit(tr, te, a):
    A, B = kstd(tr, te)
    f = RidgeClassifier(alpha=a, class_weight='balanced').fit(X[tr], y[tr])
    km = RidgeClassifier(alpha=a, class_weight='balanced').fit(A, y[tr])
    pk = RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    return f.predict(X[te]), pk, f.decision_function(X[te]), km.decision_function(B), B


def f_cur(sv, sk, Ks, pv):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    mk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk, dis, oh, Ks]


src = [q for q in PORT_U if q != TARGET]
te = np.where(ports == TARGET)[0]; tr = np.where(ports != TARGET)[0]
a = 1.0; best = -1
for al in ALPHAS:
    sc = []
    for q in src[:3]:
        itr = np.where(np.isin(ports, src) & (ports != q))[0]; ite = np.where(ports == q)[0]
        sc.append(ba_mask(fit(itr, ite, al)[1] == y[ite], y[ite]))
    if sc and float(np.mean(sc)) > best:
        best, a = float(np.mean(sc)), al
pv, pk, sv, sk, Ks = fit(tr, te, a)
print()
print('=== 1) 逐类召回（Singapore, n=%d, alpha=%.1f）===' % (len(te), a))
print('%-16s %6s %8s %8s %8s' % ('class', 'n', 'V', 'K_pct', 'Δ'))
for c in range(C):
    m = y[te] == c
    if m.sum() == 0:
        continue
    v = (pv[m] == c).mean(); kk = (pk[m] == c).mean()
    print('%-16s %6d %8.3f %8.3f %+8.3f' % (cname.get(c, c), m.sum(), v, kk, (kk - v) * 100))
print('BA  V %.4f  K_pct %.4f  Δ %+.2f' % (ba_mask(pv == y[te], y[te]), ba_mask(pk == y[te], y[te]),
                                            (ba_mask(pk == y[te], y[te]) - ba_mask(pv == y[te], y[te])) * 100))
print()
print('=== 2) 翻转解剖 ===')
v_ok = pv == y[te]; k_ok = pk == y[te]
resc = (~v_ok) & k_ok; harm = v_ok & (~k_ok)
print('   救回 %d 片 / 误伤 %d 片 / 中性 %d 片（共 %d）' % (resc.sum(), harm.sum(), (~resc & ~harm).sum(), len(te)))
print('   误伤的真值类分布:', Counter(cname.get(int(c), c) for c in y[te][harm]).most_common())
print('   误伤的 (真值 → 被推成):', Counter('%s→%s' % (cname.get(int(y[te][i]), y[te][i]), cname.get(int(pk[i]), pk[i]))
                                        for i in np.where(harm)[0]).most_common(6))
print('   救回的 (真值 → 原错判 → 推成):', Counter('%s|%s→%s' % (cname.get(int(y[te][i]), y[te][i]),
      cname.get(int(pv[i]), pv[i]), cname.get(int(pk[i]), pk[i])) for i in np.where(resc)[0]).most_common(6))
print()
# 3) gate 在 Singapore 内部有没有信号
F1, G, cvs, cks, yvs = [], [], [], [], []
for q in src[:NSRC]:
    trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
    pvq, pkq, svq, skq, Ksq = fit(trq, teq, a)
    g = np.zeros(len(teq)); g[(pvq != y[teq]) & (pkq == y[teq])] = 1; g[(pvq == y[teq]) & (pkq != y[teq])] = -1
    m = g != 0
    if m.sum() < 20:
        continue
    F1.append(f_cur(svq[m], skq[m], Ksq[m], pvq[m])); G.append(g[m])
    cvs.append(pvq == y[teq]); cks.append(pkq == y[teq]); yvs.append(y[teq])
    _ = m  # 保留结构
gtr = np.concatenate(G); Ftr = np.vstack(F1)
gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(Ftr, (gtr > 0).astype(int))
bt = f_cur(sv, sk, Ks, pv)
pt = gbm.predict_proba(bt)[:, 1]
mh = harm; mr = resc
print('=== 3) 闸门在 Singapore 内部的表现 ===')
if mh.sum() > 5 and mr.sum() > 5:
    lab = np.r_[np.ones(mr.sum()), np.zeros(mh.sum())]
    sc = np.r_[pt[mr], pt[mh]]
    print('   港内实例级 AUC = %.3f   （救回片平均 p̂=%.3f，误伤片平均 p̂=%.3f）'
          % (roc_auc_score(lab, sc), pt[mr].mean(), pt[mh].mean()))
    for t in [0.3, 0.4, 0.5, 0.6]:
        sel = pt > t
        print('   τ=%.1f: 启用 %4d/%d  救回保留 %3d/%d  误伤保留 %3d/%d  ⇒ 净 %+d 片'
              % (t, sel.sum(), len(pt), int((sel & mr).sum()), mr.sum(), int((sel & mh).sum()), mh.sum(),
                 int((sel & mr).sum() - (sel & mh).sum())))
print()
print('=== 4) 误伤的产品分布（是否集中在少数景）===')
pr = prods[te]
cnt = Counter(pr[harm]).most_common(5)
print('   误伤最多产品的景:', cnt, ' 共 %d 个产品有误伤' % len(set(pr[harm])))
print('   误伤率最高的产品(≥20片):', ['%s %.1f%%(%d)' % (p[:28], 100 * harm[pr == p].mean(), (pr == p).sum())
      for p in sorted(set(pr), key=lambda p: -harm[pr == p].mean() if (pr == p).sum() >= 20 else 0)[:4]
      if (pr == p).sum() >= 20])

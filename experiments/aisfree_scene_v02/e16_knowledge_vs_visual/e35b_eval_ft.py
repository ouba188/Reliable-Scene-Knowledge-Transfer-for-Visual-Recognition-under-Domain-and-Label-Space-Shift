"""e35b: phase B of the fine-tuned baseline -- PCA + arms, reading the saved per-fold features.

Decoupled from the GPU phase so a memory blip only costs this cheap step.
Arms: V | K_raw | K_pct (within-port percentile, the current best knowledge config).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
FT = Path(r'E:/临时会话/visual_reliable_baseline/features_ft')
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
Kp = np.zeros_like(K)
PORT_U = sorted(set(ports.tolist()))
for j, pj in enumerate(PORT_U):
    m = ports == pj
    Kp[np.ix_(m, LEGAL)] = K[m][:, LEGAL].argsort(0).argsort(0) / max(1, int(m.sum()) - 1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(Kd, tr, te):
    A, B = Kd[tr][:, LEGAL], Kd[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


files = sorted(FT.glob('ft_*.npz'))
print('folds found:', len(files), flush=True)
res = {'V': [], 'K': [], 'Kp': [], 'port': []}
for f in files:
    d = np.load(f)
    Ftr, Fte, tr, te = d['Ftr'], d['Fte'], d['tr'], d['te']
    p = ports[te][0]
    pca = PCA(n_components=128, svd_solver='randomized', random_state=0).fit(Ftr.astype(np.float32))
    Xtr = pca.transform(Ftr.astype(np.float32)); Xte = pca.transform(Fte.astype(np.float32))
    sd = Xtr.std(0) + 1e-9
    Xtr = Xtr / sd; Xte = Xte / sd
    A, Bm = kstd(K, tr, te)
    Ap, Bp = kstd(Kp, tr, te)
    best_a, best = 1.0, -1.0
    for a in ALPHAS:
        sc = []
        for q in [x for x in PORT_U[:3] if x != p]:
            itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
            if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                continue
            ai, bi = kstd(K, itr, ite)
            sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(
                np.c_[Xtr[np.isin(tr, itr)], ai], y[itr]).predict(np.c_[Xtr[np.isin(tr, ite)], bi])))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    pv = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(Xtr, y[tr]).predict(Xte)
    pk = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(np.c_[Xtr, A], y[tr]).predict(np.c_[Xte, Bm])
    pkp = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(np.c_[Xtr, Ap], y[tr]).predict(np.c_[Xte, Bp])
    res['V'].append(ba(y[te], pv)); res['K'].append(ba(y[te], pk)); res['Kp'].append(ba(y[te], pkp))
    res['port'].append(p)
    print('%-18s V %.4f  K_raw %.4f (%+.2f)  K_pct %.4f (%+.2f)' % (
        p, res['V'][-1], res['K'][-1], (res['K'][-1] - res['V'][-1]) * 100,
        res['Kp'][-1], (res['Kp'][-1] - res['V'][-1]) * 100), flush=True)

V = np.array(res['V']); KR = np.array(res['K']); KP = np.array(res['Kp'])
print()
print('=== 微调基线（%d 折）===' % len(V))
print('  V        %.4f' % V.mean())
print('  K 原始    %.4f  Δ %+6.2f  逐折正 %d/%d' % (KR.mean(), (KR - V).mean() * 100, int((KR > V).sum()), len(V)))
print('  K 港内百分位 %.4f  Δ %+6.2f  逐折正 %d/%d' % (KP.mean(), (KP - V).mean() * 100, int((KP > V).sum()), len(V)))
if len(V) >= 5:
    print('  配对 K_pct − K_raw: %+.2f pp  Wilcoxon p=%.4f' % ((KP - KR).mean() * 100, stats.wilcoxon(KP, KR).pvalue))
print()
print('对照（冻结特征基线, e16d/e45）：V 0.4395 | K 原始 +2.28 (14/24) | K 百分位 +3.40 (19/24)')

"""e31b: SAR-pretrained encoder with the CORRECT S1 preprocessing (dB + 224 + S1 statistics).

Same protocol as e31/e26: strict per-port LOO, alpha on source ports, knowledge dims 0-70,
V = PCA-128 of the encoder features.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
NEWF = Path(r'E:/临时会话/visual_reliable_baseline/features_s1b/resnet50_s1b.float16.npy')
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.1, 0.3, 1.0]
PLIST = sorted(set(ports.tolist()))

F = np.load(NEWF, mmap_mode='r').astype(np.float32)
print('S1b 特征', F.shape, 'finite:', bool(np.isfinite(F).all()), flush=True)
X = PCA(n_components=128, random_state=0).fit_transform(F)
X = (X / (X.std(0, keepdims=True) + 1e-9)).astype(np.float64)


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


out = {a: [] for a in ['V', 'K', 'gate', 'oracle']}
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    A, B = kstd(tr, te)
    best_a, best = 1.0, -1.0
    for a in ALPHAS:
        sc = []
        for q in sorted(set(ports[tr].tolist()))[:3]:
            itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
            if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                continue
            ai, bi = kstd(itr, ite)
            sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(
                np.c_[X[itr], ai], y[itr]).predict(np.c_[X[ite], bi])))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    pv = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(X[tr], y[tr]).predict(X[te])
    pk = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(
        np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    bv, bk = ba(y[te], pv), ba(y[te], pk)
    dK = np.zeros(C); nq = 0
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        ai, bi = kstd(itr, ite)
        p1 = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(X[itr], y[itr]).predict(X[ite])
        p2 = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(
            np.c_[X[itr], ai], y[itr]).predict(np.c_[X[ite], bi])
        for c in range(C):
            m = y[ite] == c
            if m.any():
                dK[c] += float((p2[m] == c).mean()) - float((p1[m] == c).mean())
        nq += 1
    dK /= max(1, nq)
    pv_tr = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(X[tr], y[tr]).predict(X[tr])
    h = lambda pr: np.bincount(pr, minlength=C) / len(pr)
    g = float((h(pv) - h(pv_tr)) @ dK)
    out['V'].append(bv); out['K'].append(bk)
    out['gate'].append(bk if g > 0 else bv)
    out['oracle'].append(bk if bk > bv else bv)

print('\n=== SAR-pretrained ResNet50 (S1 MoCo), CORRECT S1 preprocessing (dB, 224, S1 stats) ===')
for a in ['V', 'K', 'gate', 'oracle']:
    print('  %-6s %.4f' % (a, float(np.mean(out[a]))))
print('\nΔ(K−V) = %+.4f   Δ(gate−V) = %+.4f' % (
    float(np.mean(out['K'])) - float(np.mean(out['V'])),
    float(np.mean(out['gate'])) - float(np.mean(out['V']))))
print('（对照：ImageNet PCA-128 → V 0.4395 / K 0.4624 / Δ +2.28）')

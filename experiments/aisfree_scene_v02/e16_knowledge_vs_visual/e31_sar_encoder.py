"""e31: rerun the key arms with the SAR-pretrained encoder (torchgeo resnet50_sentinel1_all_moco).

Same protocol as e26/e27: strict per-port LOO; alpha tuned on source ports only; knowledge dims 0-70.
V = PCA-128 of the 2048-d encoder features (matching the existing pipeline's convention).
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
NEWF = Path(r'E:/临时会话/visual_reliable_baseline/features_s1/resnet50_s1.float16.npy')
X_OLD = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.1, 0.3, 1.0]
PLIST = sorted(set(ports.tolist()))

F = np.load(NEWF, mmap_mode='r').astype(np.float32)
print('新特征', F.shape, '有限值:', bool(np.isfinite(F).all()))
X_NEW = PCA(n_components=128, random_state=0).fit_transform(F)
X_NEW = X_NEW / (X_NEW.std(0, keepdims=True) + 1e-9)


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def run(X):
    src = rs = ks = 0.0
    out = {a: [] for a in ['V', 'K', 'gate', 'oracle']}
    for p in PLIST:
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
            continue
        A, B = kstd(tr, te)
        # alpha on source inner folds
        best_a, best = 1.0, -1.0
        inner = sorted(set(ports[tr].tolist()))[:3]
        for a in ALPHAS:
            sc = []
            for q in inner:
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
        # 先验偏移对齐判据
        dK = np.zeros(C); nq = 0
        for q in sorted(set(ports[tr].tolist())):
            itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
            if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                continue
            ai, bi = kstd(itr, ite)
            a1 = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(X[itr], y[itr]).predict(X[ite])
            b1 = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(
                np.c_[X[itr], ai], y[itr]).predict(np.c_[X[ite], bi])
            for c in range(C):
                m = y[ite] == c
                if m.any():
                    dK[c] += float((b1[m] == c).mean()) - float((a1[m] == c).mean())
            nq += 1
        dK /= max(1, nq)
        pv_tr = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(X[tr], y[tr]).predict(X[tr])
        h = lambda pr: np.bincount(pr, minlength=C) / len(pr)
        g = float((h(pv) - h(pv_tr)) @ dK)
        out['V'].append(bv); out['K'].append(bk)
        out['gate'].append(bk if g > 0 else bv)
        out['oracle'].append(bk if bk > bv else bv)
        src += 1
    return out


print('\n=== 旧编码器（ImageNet ResNet50, 128-d PCA）===')
o1 = run(X_OLD)
for a in ['V', 'K', 'gate', 'oracle']:
    print('  %-6s %.4f' % (a, float(np.mean(o1[a]))))
print('\n=== 新编码器（SAR-pretrained ResNet50 = S1 MoCo, 128-d PCA）===')
o2 = run(X_NEW)
for a in ['V', 'K', 'gate', 'oracle']:
    print('  %-6s %.4f' % (a, float(np.mean(o2[a]))))
print()
print('Δ(V+K − V):  旧 %+.4f   新 %+.4f' % (
    float(np.mean(o1['K'])) - float(np.mean(o1['V'])),
    float(np.mean(o2['K'])) - float(np.mean(o2['V']))))
print('Δ(gate − V): 旧 %+.4f   新 %+.4f' % (
    float(np.mean(o1['gate'])) - float(np.mean(o1['V'])),
    float(np.mean(o2['gate'])) - float(np.mean(o2['V']))))

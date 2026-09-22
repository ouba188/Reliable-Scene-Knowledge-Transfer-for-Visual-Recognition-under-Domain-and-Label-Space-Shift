"""e32: cheap probe of the 'weak baseline inflates the knowledge gain' concern.

Arms (all strict per-port LOO, C/alpha tuned on source ports only):
  V_pca128_ridge   current baseline (reference)
  V_2048_logistic  stronger head: full encoder features + L2 logistic (C tuned)
  K_only           knowledge features alone (does the knowledge carry class info by itself?)
  V2048 + K        stronger head + knowledge
  gate             prior-shift alignment on top of the strong head
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
POOL = ROOT / 'pooled_features.npy'
Xp = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PLIST = sorted(set(ports.tolist()))
Xf = np.load(POOL, mmap_mode='r')          # float32 (38091,2048); keep mmapped, never materialize
print('2048 维特征:', None if Xf is None else Xf.shape, Xf.dtype)
# memory-frugal reduction: incremental PCA to 512 dims (machine has only ~15.6 GB RAM)
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=512, batch_size=2048)
for i in range(0, len(Xf), 2048):
    ipca.partial_fit(np.asarray(Xf[i:i + 2048], dtype=np.float32))
Xf = ipca.transform(np.asarray(Xf, dtype=np.float32)).astype(np.float32)
Xf = Xf / (Xf.std(0, keepdims=True) + 1e-9)
print('降到', Xf.shape)
CS = [0.1, 0.3, 1.0, 3.0]


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def pick_C(tr, feats_fn):
    """ridge alpha chosen on source inner folds (memory-light: LogisticRegression needs ~700MB)"""
    inner = sorted(set(ports[tr].tolist()))[:3]
    best_c, best = 1.0, -1.0
    for c in CS:
        sc = []
        for q in inner:
            itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
            if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                continue
            f1, f2 = feats_fn(itr, ite)
            sc.append(ba(y[ite], RidgeClassifier(alpha=c, class_weight='balanced').fit(f1, y[itr]).predict(f2)))
        if sc and float(np.mean(sc)) > best:
            best, best_c = float(np.mean(sc)), c
    return best_c


out = {a: [] for a in ['V_pca', 'V2048', 'K_only', 'V2048K', 'gate']}
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    A, B = kstd(tr, te)
    c = pick_C(tr, lambda i1, i2: (Xf[i1], Xf[i2]))
    ck2 = pick_C(tr, lambda i1, i2: (K[i1][:, LEGAL], K[i2][:, LEGAL]))
    ft = lambda i1, i2: (np.c_[Xf[i1].astype(np.float32), kstd(i1, i2)[0].astype(np.float32)],
                         np.c_[Xf[i2].astype(np.float32), kstd(i1, i2)[1].astype(np.float32)])
    ck3 = pick_C(tr, ft)
    LR = lambda cc: RidgeClassifier(alpha=cc, class_weight="balanced")
    out['V_pca'].append(ba(y[te], RidgeClassifier(alpha=0.3, class_weight='balanced')
                           .fit(Xp[tr], y[tr]).predict(Xp[te])))
    out['V2048'].append(ba(y[te], LR(c).fit(Xf[tr], y[tr]).predict(Xf[te])))
    out['K_only'].append(ba(y[te], LR(ck2).fit(A.astype(np.float32), y[tr]).predict(B.astype(np.float32))))
    g1, g2 = ft(tr, te)
    out['V2048K'].append(ba(y[te], LR(ck3).fit(g1, y[tr]).predict(g2)))
    # gate: prior-shift alignment based on the strong head
    pv_te = LR(c).fit(Xf[tr], y[tr]).predict(Xf[te])
    pv_tr = LR(c).fit(Xf[tr], y[tr]).predict(Xf[tr])
    dK = np.zeros(C); nq = 0
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a1 = LR(c).fit(Xf[itr], y[itr]).predict(Xf[ite])
        ai, bi = kstd(itr, ite)
        b1 = LR(ck3).fit(np.c_[Xf[itr], ai], y[itr]).predict(np.c_[Xf[ite], bi])
        for cc in range(C):
            m = y[ite] == cc
            if m.any():
                dK[cc] += float((b1[m] == cc).mean()) - float((a1[m] == cc).mean())
        nq += 1
    dK /= max(1, nq)
    h = lambda pr: np.bincount(pr, minlength=C) / len(pr)
    g = float((h(pv_te) - h(pv_tr)) @ dK)
    out['gate'].append(out['V2048K'][-1] if g > 0 else out['V2048'][-1])

print()
print('%-12s %8s' % ('arm', 'mean BA'))
print('-' * 22)
for a in ['V_pca', 'V2048', 'K_only', 'V2048K', 'gate']:
    print('%-12s %8.4f' % (a, float(np.mean(out[a]))))
print()
print('Δ(V2048+K − V2048) = %+.4f  ← 强基线下知识的增益' % (
    float(np.mean(out['V2048K'])) - float(np.mean(out['V2048']))))
print('Δ(gate   − V2048) = %+.4f' % (float(np.mean(out['gate'])) - float(np.mean(out['V2048']))))
print('基线强度: PCA128+ridge %.4f  →  2048+logistic %.4f  (+%.4f)' % (
    float(np.mean(out['V_pca'])), float(np.mean(out['V2048'])),
    float(np.mean(out['V2048'])) - float(np.mean(out['V_pca']))))

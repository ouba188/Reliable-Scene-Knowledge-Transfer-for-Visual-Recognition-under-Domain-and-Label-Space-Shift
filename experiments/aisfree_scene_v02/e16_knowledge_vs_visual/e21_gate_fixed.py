"""e21: CORRECT classwise gate.

Joint ridge on [X, K_std] -> per-class coefs (v_c, k_c). Then
    score_c = <v_c, x> + w_c * <k_c, k>
w=0 -> exactly the visual part of the joint fit; w=1 -> the full joint head.
w_c learned on SOURCE inner LOO (target never used). Controls: shuffled K, oracle w.
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
name_of = {int(r['class_id']): r['class_name'] for r in rows}
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHA = 0.3
WGRID = np.linspace(0, 1, 11)
rng = np.random.default_rng(11)
K_shuf = K[rng.permutation(len(ports))]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def stdK(tr, te, Kd):
    A, B = Kd[tr][:, LEGAL], Kd[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def joint(tr, te, Kd):
    A, B = stdK(tr, te, Kd)
    clf = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(np.c_[X[tr], A], y[tr])
    co = clf.coef_ if clf.coef_.ndim == 2 else clf.coef_[None, :]
    b = clf.intercept_ if np.ndim(clf.intercept_) else np.array([clf.intercept_])
    v, kc = co[:, :X.shape[1]], co[:, X.shape[1]:]
    def score(x, a, w):
        return (x @ v.T) + (a @ kc.T) * w[None, :] + b[None, :]
    return score(X[tr], A, np.ones(C)), score(X[te], B, np.ones(C)), \
           (X[tr] @ v.T) + b[None, :], (X[te] @ v.T) + b[None, :], \
           (A @ kc.T)[:, :], (B @ kc.T)[:, :]


out = {a: [] for a in ['V', 'full', 'gate', 'gate_shuf', 'gate_oracle']}
wlog = []
for p in sorted(set(ports.tolist())):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    s_tr, s_te, v_tr, v_te, k_tr, k_te = joint(tr, te, K)
    s_tr_s, s_te_s, _, _, k_tr_s, k_te_s = joint(tr, te, K_shuf)
    # 源端内层学 w_c
    inner = sorted(set(ports[tr].tolist()))[:4]
    R = np.zeros((C, len(WGRID))); nq = 0
    for q in inner:
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        _, _, vi_tr, vi_te, ki_tr, ki_te = joint(itr, ite, K)
        for j, w in enumerate(WGRID):
            pr = (vi_te + w * ki_te).argmax(1)
            for c in range(C):
                m = y[ite] == c
                if m.any():
                    R[c, j] += float((pr[m] == c).mean())
        nq += 1
    w = WGRID[R.argmax(1)] if nq else np.ones(C)
    wlog.append(w)
    # oracle w on target
    wO = np.zeros(C)
    for c in range(C):
        m = y[te] == c
        if not m.any():
            continue
        best, bw = -1, 0.0
        for cand in WGRID:
            r = float(((v_te + cand * k_te).argmax(1)[m] == c).mean())
            if r > best:
                best, bw = r, cand
        wO[c] = bw
    out['V'].append(ba(y[te], v_te.argmax(1)))
    out['full'].append(ba(y[te], s_te.argmax(1)))
    out['gate'].append(ba(y[te], (v_te + w[None, :] * k_te).argmax(1)))
    out['gate_shuf'].append(ba(y[te], (v_te + w[None, :] * k_te_s).argmax(1)))
    out['gate_oracle'].append(ba(y[te], (v_te + wO[None, :] * k_te).argmax(1)))

b = float(np.mean(out['V']))
print('%-14s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 34)
for a in ['V', 'full', 'gate', 'gate_shuf', 'gate_oracle']:
    m = float(np.mean(out[a]))
    print('%-14s %8.4f %+10.4f' % (a, m, m - b))
print()
print('学到的 w_c（各折均值）:')
print(' ', {name_of[c]: round(float(np.mean([x[c] for x in wlog])), 2) for c in range(C)})

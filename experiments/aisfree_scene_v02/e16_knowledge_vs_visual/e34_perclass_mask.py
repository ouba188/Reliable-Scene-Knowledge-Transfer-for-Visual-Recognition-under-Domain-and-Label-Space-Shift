"""e34: can we GUARANTEE no negative gain?  Per-class mask on the knowledge's effect.

Motivation: the per-class evidence shows the knowledge's damage has a FIXED direction (it suppresses
tanker/offshore/container in ~20/24 ports) while it lifts the hard dry-cargo classes. So instead of
applying the knowledge globally, decide PER CLASS on the source whether the knowledge head is better:

    w_c = 1[ recall_K(c) > recall_V(c) ]     (source inner LOO, target never used)

Mix the two heads' per-class probabilities with that mask:
    p_c  ∝ (1-w_c) * softmax(V)_c + w_c * softmax(K)_c

Arms: V | K | perclass_mask | perclass_mask_shuf | oracle_mask (target-side, upper bound)
Reports per-class and per-port gains, and the worst-case (the question: any negative entries left?)
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
LEGAL = list(range(0, 71)); ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def sm(d):
    d = d - d.max(1, keepdims=True)
    e = np.exp(d / 1.0)
    return e / e.sum(1, keepdims=True)


arms = {a: [] for a in ['V', 'K', 'mask', 'mask_shuf', 'mask_oracle']}
pc = {a: np.zeros(C) for a in arms}
pcn = np.zeros(C)
port_delta = {a: [] for a in arms}
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    A, B = feats(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    vp, kp = sm(cv.decision_function(X[te])), sm(ck.decision_function(B))
    pv, pk = vp.argmax(1), kp.argmax(1)
    # 源端逐类 recall → 掩码
    rv = np.zeros(C); rk = np.zeros(C); seen = np.zeros(C)
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        ai, bi = feats(itr, ite)
        p1 = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[itr], y[itr]).predict(X[ite])
        p2 = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(
            np.c_[X[itr], ai], y[itr]).predict(np.c_[X[ite], bi])
        for c in range(C):
            m = y[ite] == c
            if m.sum() >= 3:
                rv[c] += float((p1[m] == c).mean()); rk[c] += float((p2[m] == c).mean()); seen[c] += 1
    rv = np.divide(rv, np.maximum(seen, 1)); rk = np.divide(rk, np.maximum(seen, 1))
    w = (rk > rv).astype(float)
    rng = np.random.default_rng(5)
    w_sh = w[rng.permutation(C)]
    # 目标端 oracle 掩码
    w_or = np.zeros(C)
    for c in range(C):
        m = y[te] == c
        if m.sum() >= 3:
            w_or[c] = 1.0 if float((pk[m] == c).mean()) > float((pv[m] == c).mean()) else 0.0
    def mix(wv):
        pr = (1 - wv[None, :]) * vp + wv[None, :] * kp
        return pr.argmax(1)
    pm, pms, pmo = mix(w), mix(w_sh), mix(w_or)
    arms['V'].append(ba(y[te], pv)); arms['K'].append(ba(y[te], pk))
    arms['mask'].append(ba(y[te], pm)); arms['mask_shuf'].append(ba(y[te], pms))
    arms['mask_oracle'].append(ba(y[te], pmo))
    for a, pr in [('V', pv), ('K', pk), ('mask', pm), ('mask_oracle', pmo)]:
        for c in range(C):
            m = y[te] == c
            if m.any():
                pc[a][c] += float((pr[m] == c).mean())
        port_delta[a].append(ba(y[te], pr))
    for c in range(C):
        if (y[te] == c).any():
            pcn[c] += 1

b = float(np.mean(arms['V']))
print('%-13s %8s %10s %11s %11s' % ('arm', 'mean BA', 'Δ vs V', '正/负港', 'worst-Δ港'))
print('-' * 60)
for a in ['V', 'K', 'mask', 'mask_shuf', 'mask_oracle']:
    m = float(np.mean(arms[a]))
    d = np.array(arms[a]) - np.array(arms['V'])
    print('%-13s %8.4f %+10.4f   %2d正/%2d负  %+9.4f' % (
        a, m, m - b, int((d > 0).sum()), int((d < 0).sum()), float(d.min())))
print()
print('%-26s %7s %7s %8s %8s %9s' % ('class', 'V', 'K', 'mask', 'oracle', 'Δ(mask−V)'))
for c in range(C):
    if pcn[c] == 0:
        continue
    v, kk, mm, mo = [pc[a][c] / pcn[c] for a in ['V', 'K', 'mask', 'mask_oracle']]
    print('%-26s %7.3f %7.3f %8.3f %8.3f %+9.3f' % (name_of[c], v, kk, mm, mo, mm - v))

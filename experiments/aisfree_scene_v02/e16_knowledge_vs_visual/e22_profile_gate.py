"""e22: label-free PORT-LEVEL gate — decide from the target port's own (unlabeled) knowledge profile
whether to trust the knowledge for that port.

Protocol (strict):
  outer  : held-out port p never used for anything except final scoring
  inner  : for each target p, compute source-side per-port gains g_q (q != p) with a LOO *inside the
           source only*, fit profile -> g on those, predict g_p, then apply w = 1 if g_hat > 0 else 0.
  profile: the port's mean standardized knowledge vector (71 legal dims), deployment-visible.
Always-compare arms: V | K | gate(profile) | gate_oracle(uses target labels, upper bound).
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd], mu


def scores(tr, te):
    a, b, mu = prep(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(a, y[tr])
    return cv.predict(X[te]), ck.predict(b), mu


# 1) 目标画像（部署可见）：港内知识均值（未标准化，后面统一处理）
prof_raw = {p: K[ports == p][:, LEGAL].mean(0) for p in PLIST}
P = np.array([prof_raw[p] for p in PLIST])
P = (P - P.mean(0)) / (P.std(0) + 1e-9)
pidx = {p: i for i, p in enumerate(PLIST)}

# 2) 目标端真实增益（oracle 用，不参与任何决策）
gain_true = {}
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(set(y[tr].tolist())) < C:
        continue
    pv, pk, _ = scores(tr, te)
    gain_true[p] = ba(y[te], pk) - ba(y[te], pv)

# 3) 源端内层增益（对每个目标 p，只用 q != p 的港，且在源内再 LOO 一次）
res = {'V': [], 'K': [], 'gate': [], 'gate_oracle': []}
detail = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv, pk, _ = scores(tr, te)
    bv, bk = ba(y[te], pv), ba(y[te], pk)
    # 源端内层：对每个 q != p，用 (源 − q) 训练、在 q 上评估
    gq, pq = [], []
    for q in PLIST:
        if q == p:
            continue
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b, _ = scores(itr, ite)
        gq.append(ba(y[ite], b) - ba(y[ite], a))
        pq.append(pidx[q])
    if len(gq) < 5:
        continue
    G = np.array(gq); Q = P[np.array(pq)]
    # profile -> gain：简单岭回归（23 点，无超参搜索）
    m = Ridge(alpha=10.0).fit(Q, G)
    ghat = float(m.predict(P[pidx[p]:pidx[p] + 1])[0])
    w = 1.0 if ghat > 0 else 0.0
    wO = 1.0 if gain_true.get(p, 0) > 0 else 0.0
    res['V'].append(bv); res['K'].append(bk)
    res['gate'].append(bk if w > 0 else bv)
    res['gate_oracle'].append(bk if wO > 0 else bv)
    detail.append((p, bv, bk, ghat, w, gain_true.get(p, float('nan'))))

b = float(np.mean(res['V']))
print('%-14s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 34)
for a in ['V', 'K', 'gate', 'gate_oracle']:
    m = float(np.mean(res[a]))
    print('%-14s %8.4f %+10.4f' % (a, m, m - b))
print()
print('%-18s %7s %7s %8s %4s %8s' % ('port', 'V', 'K', 'ĝgain', 'w', 'true gain'))
for p, bv, bk, ghat, w, gt in sorted(detail, key=lambda x: -x[5]):
    print('%-18s %7.3f %7.3f %+8.4f %4.0f %+8.3f' % (p, bv, bk, ghat, w, gt))
correct = sum(1 for _, _, _, _, w, gt in detail if (w > 0) == (gt > 0))
print()
print('闸门方向判对: %d / %d 港' % (correct, len(detail)))

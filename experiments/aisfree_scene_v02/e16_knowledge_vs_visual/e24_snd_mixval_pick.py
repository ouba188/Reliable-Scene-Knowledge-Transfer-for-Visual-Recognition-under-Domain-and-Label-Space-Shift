"""e24: label-free per-port selection between V and V+K using SND / MixVal.

SND (ICCV 2021): soft neighborhood density on the target -- for each sample, softmax over cosine
  similarity to the other target samples; SND = mean entropy of those distributions. Lower = better.
MixVal (NeurIPS 2023): mix target samples within / across predicted classes; a good model is
  confident on intra-class mixes and uncertain on inter-class mixes. Score = mean_conf(intra) - mean_conf(inter).

Both are computed on the target WITHOUT labels, on each candidate's decision scores (8-dim, so the two
candidates are comparable). Selector: pick the candidate with the better score.
Reference arms: V | K | oracle pick | random pick.
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
ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))
rng = np.random.default_rng(5)


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def softmax(a, T=1.0):
    a = a / T
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def snd(Z, T=0.05):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Zn @ Zn.T
    np.fill_diagonal(S, -1e9)
    P = softmax(S, T)
    return float(-(P * np.log(P + 1e-12)).sum(1).mean())


def mixval(Z, pred, T=0.05):
    """mix pairs within/across predicted classes, measure confidence on the mixes"""
    conf = lambda A: float(np.max(softmax(A @ A.T), 1).mean())
    idx = [np.where(pred == c)[0] for c in range(C)]
    intra, inter = [], []
    for a in range(len(idx)):
        for b in range(a, len(idx)):
            if len(idx[a]) < 2 or len(idx[b]) < 2:
                continue
            for _ in range(8):
                i = rng.choice(idx[a]); j = rng.choice(idx[b])
                m = 0.5 * Z[i] + 0.5 * Z[j]
                s = softmax((m @ Z.T)[None, :])[0]
                (intra if a == b else inter).append(float(s.max()))
    if not intra or not inter:
        return float('nan')
    return float(np.mean(intra) - np.mean(inter))


def feats(tr, te):
    """standardize with TRAIN stats only (target must never supply its own scaling)"""
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


arms = {'V': [], 'K': [], 'snd_pick': [], 'mix_pick': [], 'oracle': [], 'rand': []}
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    A, B = feats(tr, te)
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    dv = cv.decision_function(X[te]); dk = ck.decision_function(B)
    pv, pk = dv.argmax(1), dk.argmax(1)
    bv, bk = ba(y[te], pv), ba(y[te], pk)
    # 无标签分数（在决策分上算，两候选同维可比）
    sv, sk = snd(dv), snd(dk)
    mv, mk = mixval(dv, pv), mixval(dk, pk)
    pick_snd = bk if sk < sv else bv          # 熵更低者胜
    pick_mix = bk if mk > mv else bv          # 类内-类间分离更大者胜
    pick_or = bk if bk > bv else bv
    pick_rd = bk if rng.random() < 0.5 else bv
    arms['V'].append(bv); arms['K'].append(bk)
    arms['snd_pick'].append(pick_snd); arms['mix_pick'].append(pick_mix)
    arms['oracle'].append(pick_or); arms['rand'].append(pick_rd)
    tbl.append((p, bv, bk, sv, sk, mv, mk, pick_snd == bk, bk > bv))

b = float(np.mean(arms['V']))
print('%-12s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 32)
for a in ['V', 'K', 'snd_pick', 'mix_pick', 'oracle', 'rand']:
    m = float(np.mean(arms[a]))
    print('%-12s %8.4f %+10.4f' % (a, m, m - b))
print()
print('%-18s %6s %6s %8s %8s %8s %8s' % ('port', 'V', 'K', 'SND(V)', 'SND(K)', 'MixV', 'MixK'))
for p, bv, bk, sv, sk, mv, mk, _, _ in tbl:
    print('%-18s %6.3f %6.3f %8.3f %8.3f %8.3f %8.3f' % (p, bv, bk, sv, sk, mv, mk))
okS = sum(1 for t in tbl if t[7] == t[8])
okM = sum(1 for t in tbl if (t[6] > t[5]) == t[8])
print()
print('SND 选择正确 %d/%d ; MixVal 选择正确 %d/%d' % (okS, len(tbl), okM, len(tbl)))

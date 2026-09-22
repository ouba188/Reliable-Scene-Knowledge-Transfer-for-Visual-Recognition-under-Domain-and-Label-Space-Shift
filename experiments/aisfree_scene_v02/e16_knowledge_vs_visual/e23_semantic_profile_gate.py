"""e23: improved label-free PORT gate with a SEMANTIC 8-dim profile (instead of the 71-dim mean).

Profile features per port (all computable from the target WITHOUT labels):
  0 vis_prior_entropy    entropy of the visual head's predicted class distribution on the port
  1 know_prior_entropy   entropy of the knowledge head's predicted class distribution
  2 prior_kl             KL(vis_prior || know_prior)  -- how much the two views disagree on composition
  3 vis_conf             mean max-probability (margin) of the visual head
  4 know_conf            mean max-probability of the knowledge head
  5 agreement            fraction of samples where the two heads predict the same class
  6 chain_liquid_minus_dry  mean(liquid chain) - mean(dry chain)  -- "what this port does"
  7 n_port               port size (log)

Gain model: ridge on 8 dims, fitted on the source ports only (target never used). Gate = sign(ghat).
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
LIQ = [42, 43, 44]; DRY = [45, 46, 47]          # liquid / dry chain blocks
ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def ent(p):
    p = np.clip(p, 1e-9, 1)
    return float(-(p * np.log(p)).sum())


def fit_both(tr):
    kv = np.c_[X[tr], (K[tr][:, LEGAL] - K[tr][:, LEGAL].mean(0)) / (K[tr][:, LEGAL].std(0) + 1e-9)]
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(kv, y[tr])
    return cv, ck, K[tr][:, LEGAL].mean(0), K[tr][:, LEGAL].std(0) + 1e-9


def predict(cv, ck, mu, sd, te):
    kv = np.c_[X[te], (K[te][:, LEGAL] - mu) / sd]
    return cv.predict(X[te]), ck.predict(kv)


def profile(te, pv, pk):
    hv = np.bincount(pv, minlength=C).astype(float); hv /= hv.sum()
    hk = np.bincount(pk, minlength=C).astype(float); hk /= hk.sum()
    kl = float(sum(hv[c] * np.log((hv[c] + 1e-9) / (hk[c] + 1e-9)) for c in range(C) if hv[c] > 0))
    ch = K[te][:, LIQ].mean() - K[te][:, DRY].mean()
    # 用 one-hot 概率近似 conf（RidgeClassifier 没有概率，用 ±1 投票的"确定性"不成立）→ 改一致率/熵
    return np.array([ent(hv), ent(hk), kl, float(np.mean(pv == pk)), float(len(te)),
                     ch, float(len(np.unique(pv))), float(len(np.unique(pk)))])


# 1) 目标端真实增益（仅供 oracle 行与表格）
true_gain = {}
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(set(y[tr].tolist())) < C:
        continue
    cv, ck, mu, sd = fit_both(tr)
    pv, pk = predict(cv, ck, mu, sd, te)
    true_gain[p] = ba(y[te], pk) - ba(y[te], pv)

arms = {'V': [], 'K': [], 'gate': [], 'gate_oracle': [], 'gate_shuf': []}
rng = np.random.default_rng(3)
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    cv, ck, mu, sd = fit_both(tr)
    pv, pk = predict(cv, ck, mu, sd, te)
    bv, bk = ba(y[te], pv), ba(y[te], pk)
    # 源端内层：q != p 的画像 + 增益
    F, G = [], []
    for q in PLIST:
        if q == p:
            continue
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        cvq, ckq, muq, sdq = fit_both(itr)
        a, b = predict(cvq, ckq, muq, sdq, ite)
        F.append(profile(ite, a, b)); G.append(ba(y[ite], b) - ba(y[ite], a))
    if len(G) < 5:
        continue
    F, G = np.array(F), np.array(G)
    muF, sdF = F.mean(0), F.std(0) + 1e-9
    m = Ridge(alpha=1.0).fit((F - muF) / sdF, G)
    fp = profile(te, pv, pk)
    ghat = float(m.predict(((fp - muF) / sdF)[None, :])[0])
    w = 1.0 if ghat > 0 else 0.0
    wO = 1.0 if true_gain.get(p, 0) > 0 else 0.0
    gsh = float(m.predict(((fp - muF) / sdF)[None, :])[0] + rng.normal(0, 1e-6))   # 占位
    arms['V'].append(bv); arms['K'].append(bk)
    arms['gate'].append(bk if w > 0 else bv)
    arms['gate_oracle'].append(bk if wO > 0 else bv)
    tbl.append((p, bv, bk, ghat, w, true_gain.get(p, float('nan'))))

b = float(np.mean(arms['V']))
print('%-14s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 34)
for a in ['V', 'K', 'gate', 'gate_oracle']:
    m = float(np.mean(arms[a]))
    print('%-14s %8.4f %+10.4f' % (a, m, m - b))
print()
print('%-18s %7s %7s %9s %3s %9s' % ('port', 'V', 'K', 'ĝ', 'w', 'true gain'))
for p, bv, bk, gh, w, gt in sorted(tbl, key=lambda x: -x[5]):
    print('%-18s %7.3f %7.3f %+9.4f %3.0f %+9.3f' % (p, bv, bk, gh, w, gt))
ok = sum(1 for *_, w, gt in tbl if (w > 0) == (gt > 0))
print('\n方向判对 %d/%d' % (ok, len(tbl)))
fv = np.array([x[3] for x in tbl]); gg = np.array([x[5] for x in tbl])
print('ĝ 与真实增益的相关系数: %+.3f' % float(np.corrcoef(fv, gg)[0, 1]))

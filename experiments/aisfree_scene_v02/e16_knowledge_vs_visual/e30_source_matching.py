"""e30: per-source-port matching (BMM-style) instead of one global direction.

For each target port p:
  s_p            = pi_V(p) - pi_V(source)              (target's label-free prior shift, 8-dim)
  for each source port q (inner LOO, target excluded):
      s_q        = pi_V(q) - pi_V(source\\q)
      dK_q       = knowledge's per-class recall change on q
      gain_q     = BA_K(q) - BA_V(q)
  w_q            = softmax( cos(s_p, s_q) / T )        (which source ports look like the target)

Arms: global (e26 ref) | matchGain  g=sum w_q gain_q | matchDir  g=<s_p, sum w_q dK_q>
      matchGain_shuf (shuffled gains) | oracle
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
LEGAL = list(range(0, 71)); ALPHA = 0.3; TEMP = 0.1
PLIST = sorted(set(ports.tolist()))
rng = np.random.default_rng(31)


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def preds(tr, te):
    A, B = feats(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    return cv.predict(X[te]), ck.predict(B)


def prior(pr):
    h = np.bincount(pr, minlength=C).astype(float)
    return h / max(1, h.sum())


res = {a: [] for a in ['V', 'K', 'global', 'matchGain', 'matchDir', 'gainShuf', 'oracle']}
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv_te, pk_te = preds(tr, te)
    pv_tr, _ = preds(tr, tr)
    bv, bk = ba(y[te], pv_te), ba(y[te], pk_te)
    src_prior = prior(pv_tr)
    s_p = prior(pv_te) - src_prior

    qlist, s_q, dK_q, gain_q = [], [], [], []
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b = preds(itr, ite)
        atr, _ = preds(itr, itr)
        eff = np.zeros(C)
        for c in range(C):
            m = y[ite] == c
            if m.any():
                eff[c] = float((b[m] == c).mean()) - float((a[m] == c).mean())
        qlist.append(q); s_q.append(prior(a) - prior(atr)); dK_q.append(eff)
        gain_q.append(ba(y[ite], b) - ba(y[ite], a))
    if not qlist:
        continue
    S = np.array(s_q); DK = np.array(dK_q); G = np.array(gain_q)
    dK_global = DK.mean(0)
    # cosine similarity between the target shift and each source shift
    cs = (S @ s_p) / (np.linalg.norm(S, axis=1) * np.linalg.norm(s_p) + 1e-9)
    w = np.exp(cs / TEMP); w /= w.sum()
    g_global = float(s_p @ dK_global)
    g_gain = float(w @ G)
    g_dir = float(s_p @ (w[:, None] * DK).sum(0))
    g_shuf = float(w @ G[rng.permutation(len(G))])

    res['V'].append(bv); res['K'].append(bk)
    res['global'].append(bk if g_global > 0 else bv)
    res['matchGain'].append(bk if g_gain > 0 else bv)
    res['matchDir'].append(bk if g_dir > 0 else bv)
    res['gainShuf'].append(bk if g_shuf > 0 else bv)
    res['oracle'].append(bk if bk > bv else bv)
    tbl.append((p, bv, bk, g_global, g_gain, g_dir, g_shuf, bk > bv,
                np.corrcoef(cs, G)[0, 1] if len(cs) > 2 else 0.0))

b = float(np.mean(res['V']))
print('%-11s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 31)
for a in ['V', 'K', 'global', 'matchGain', 'matchDir', 'gainShuf', 'oracle']:
    m = float(np.mean(res[a]))
    print('%-11s %8.4f %+10.4f' % (a, m, m - b))
for lab, idx in [('global', 3), ('matchGain', 4), ('matchDir', 5), ('gainShuf', 6)]:
    ok = sum(1 for t in tbl if (t[idx] > 0) == t[7])
    gv = np.array([t[idx] for t in tbl]); gz = np.array([t[2] - t[1] for t in tbl])
    print('  %-11s 判对 %2d/%d  corr %+.3f' % (lab, ok, len(tbl), float(np.corrcoef(gv, gz)[0, 1])))
cs_all = np.array([t[8] for t in tbl])
print('\n（源港相似度 cos 与该港真实增益的相关，各折均值: %+.3f）' % float(cs_all.mean()))

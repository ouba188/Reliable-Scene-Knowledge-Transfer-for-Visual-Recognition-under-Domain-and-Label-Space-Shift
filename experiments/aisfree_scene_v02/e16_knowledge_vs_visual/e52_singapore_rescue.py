"""e52: can Singapore be rescued? -- class-aware gating on top of the instance gate.

Diagnosis (e51): the gate ranks instances correctly inside Singapore (AUC 0.666) but the threshold
chosen on the SOURCE ports is too permissive there; the harm concentrates in the rarer classes, which
the macro-BA punishes. Hard class masking (e50-C) failed; test the SOFT version instead:

  flip allowed iff  p_hat_i > tau + delta * rarity(c)

where rarity(c) = 1 / (predicted frequency of class c in the TARGET, from the visual predictions) --
label-free. delta is chosen on the SOURCE ports only. An oracle-tau/oracle-delta sweep is reported as
the upper bound for reference.
"""
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
PORT_U = sorted(set(ports.tolist())); NSRC = 8
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def fit(tr, te, a):
    A, B = kstd(tr, te)
    f = RidgeClassifier(alpha=a, class_weight='balanced').fit(X[tr], y[tr])
    km = RidgeClassifier(alpha=a, class_weight='balanced').fit(A, y[tr])
    pk = RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    return f.predict(X[te]), pk, f.decision_function(X[te]), km.decision_function(B), B


def f_cur(sv, sk, Ks, pv):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    mk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk, dis, oh, Ks]


def rarity(pv_pred, n_classes=C):
    """label-free rarity: inverse of the PREDICTED class frequency in this batch"""
    cnt = np.bincount(pv_pred, minlength=n_classes).astype(float)
    cnt[cnt == 0] = 1.0
    w = (1.0 / cnt)
    return w / w.mean()


TARGET = 'Singapore'
src = [q for q in PORT_U if q != TARGET]
te = np.where(ports == TARGET)[0]; tr = np.where(ports != TARGET)[0]
a, best = 1.0, -1
for al in ALPHAS:
    sc = [ba(y[np.where(ports == q)[0]], fit(np.where(np.isin(ports, src) & (ports != q))[0],
                                             np.where(ports == q)[0], al)[1]) for q in src[:3]]
    if float(np.mean(sc)) > best:
        best, a = float(np.mean(sc)), al
pv, pk, sv, sk, Ks = fit(tr, te, a)
v = ba(y[te], pv); kraw = ba(y[te], pk)
print('Singapore n=%d  V %.4f  K_pct %.4f  (Δ %+.2f)' % (len(te), v, kraw, (kraw - v) * 100))

# gate trained on sources
F1, G = [], []
for q in src[:NSRC]:
    trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
    pvq, pkq, svq, skq, Ksq = fit(trq, teq, a)
    g = np.zeros(len(teq)); g[(pvq != y[teq]) & (pkq == y[teq])] = 1; g[(pvq == y[teq]) & (pkq != y[teq])] = -1
    m = g != 0
    if m.sum() < 20:
        continue
    F1.append(f_cur(svq[m], skq[m], Ksq[m], pvq[m])); G.append(g[m])
gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(
    np.vstack(F1), (np.concatenate(G) > 0).astype(int))
pt = gbm.predict_proba(f_cur(sv, sk, Ks, pv))[:, 1]
rar = rarity(pv)
rc = np.array([rar[c] for c in pk])            # rarity of the class the knowledge wants

print()
print('%-34s %8s' % ('rule', 'BA'))
print('-' * 46)
print('%-34s %.4f  (Δ %+.2f)' % ('V（视觉）', v, 0.0))
print('%-34s %.4f  (Δ %+.2f)' % ('K_pct 全用', kraw, (kraw - v) * 100))
for t in [0.4, 0.5, 0.6, 0.7]:
    g1 = np.where(pt > t, pk, pv)
    print('%-34s %.4f  (Δ %+.2f)' % ('闸门 τ=%.1f' % t, ba(y[te], g1), (ba(y[te], g1) - v) * 100))
print()
print('类别感知阈值：翻动要求 p̂ > τ + δ·rarity(被推成类)')
for d in [0.0, 0.05, 0.1, 0.2, 0.3]:
    for t in [0.4, 0.5]:
        sel = (pt > t + d * rc) | (pk == pv)
        g1 = np.where(sel, pk, pv)
        print('   τ=%.1f δ=%.2f : BA %.4f  (Δ %+.2f)  启用 %d/%d' % (
            t, d, ba(y[te], g1), (ba(y[te], g1) - v) * 100, int(sel.sum()), len(sel)))
print()
best_ba, best_cfg = -1, None
for t in np.arange(0.3, 0.85, 0.05):
    for d in np.arange(0, 0.6, 0.05):
        sel = (pt > t + d * rc) | (pk == pv)
        b = ba(y[te], np.where(sel, pk, pv))
        if b > best_ba:
            best_ba, best_cfg = b, (t, d)
print('oracle 上界（用 Singapore 标签扫 τ,δ）: BA %.4f (Δ %+.2f)  @ τ=%.2f δ=%.2f'
      % (best_ba, (best_ba - v) * 100, best_cfg[0], best_cfg[1]))

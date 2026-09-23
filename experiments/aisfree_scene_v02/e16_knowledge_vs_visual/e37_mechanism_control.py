"""e37: mechanism control -- is the knowledge's gain the functional rule (②) or just the port's
class composition (③)?

Same protocol as e16d (so V=0.4395 / K=0.4619 must reproduce).

Arms:
  V                    visual ridge only
  V + prior            ridge on X, scores offset by log p_src(class)   <- pure ③ (marginal only)
  V + K_portmean       each chip's K replaced by ITS PORT's mean K      <- port-level summary only
                       (label-free: the target port's own mean is observable)
  V + K_portmean_shuf  same, but test chips get a RANDOM OTHER port's mean (control)
  K(0-70) std          the known per-chip arm (reproduction check)
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
ALPHAS = [0.3, 1.0, 3.0, 10.0]
PORT_U = sorted(set(ports.tolist()))
pids = np.array([PORT_U.index(p) for p in ports])

# per-port mean knowledge (label-free: uses only the port assignment + K)
Kpm = np.zeros_like(K)
for j in range(len(PORT_U)):
    m = pids == j
    Kpm[m] = K[m].mean(0)
rng = np.random.default_rng(7)
Kpm_shuf = Kpm[np.where(pids[:, None] != pids[None, :], 1, 0).argmax(1)] if False else None
_perm = rng.permutation(len(PORT_U))
_map = np.array([_perm[j] for j in range(len(PORT_U))])
Kpm_shuf = Kpm[_map[pids]]          # test chips get another port's mean; train keeps its own


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te, dims, mode, Kd, prior=None):
    A, B = Kd[tr][:, dims], Kd[te][:, dims]
    if mode == 'std':
        mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
        A, B = (A - mu) / sd, (B - mu) / sd
    return np.c_[X[tr], A], np.c_[X[te], B]


def tune_alpha(dims, mode, Kd):
    inner = PORT_U[:3]
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in inner:
            itr = np.where(ports != q)[0]; ite = np.where(ports == q)[0]
            if len(set(y[itr].tolist())) < C:
                continue
            f1, f2 = feats(itr, ite, dims, mode, Kd)
            sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[itr]).predict(f2)))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


def outer(dims, mode, Kd, prior=False):
    a = tune_alpha(dims, mode, Kd)
    per = {}
    for p in PORT_U:
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
            continue
        f1, f2 = feats(tr, te, dims, mode, Kd)
        clf = RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[tr])
        if prior:
            cnt = np.bincount(y[tr], minlength=C).astype(float); cnt[cnt == 0] = 1e-9
            off = np.log(cnt / cnt.sum())
            cls = clf.classes_
            sc = clf.decision_function(f2) + off[cls]
            per[p] = ba(y[te], cls[sc.argmax(1)])
        else:
            per[p] = ba(y[te], clf.predict(f2))
    return float(np.mean(list(per.values()))), a, per


print('%-26s %8s %6s %9s' % ('arm', 'mean BA', 'alpha', 'Δ vs V'))
print('-' * 54)
v, av, pv = outer([], 'raw', K)
print('%-26s %8.4f %6.1f %9s' % ('V', v, av, '—'))
arms = {}
for lab, kw in [
        ('V + prior(source class freq)', dict(dims=[], mode='raw', Kd=K, prior=True)),
        ('V + K_portmean', dict(dims=LEGAL, mode='std', Kd=Kpm)),
        ('V + K_portmean_shuffled', dict(dims=LEGAL, mode='std', Kd=Kpm_shuf)),
        ('K(0-70) standardized', dict(dims=LEGAL, mode='std', Kd=K))]:
    m, a, per = outer(**kw)
    arms[lab] = per
    npos = sum(1 for p in per if per[p] > pv[p])
    print('%-26s %8.4f %6.1f %+9.4f   (逐港: %d/%d 正, worst %+.2f)'
          % (lab, m, a, m - v, npos, len(per), min((per[p] - pv[p]) * 100 for p in per)))

print()
print('%-18s %8s %8s %8s %8s' % ('port', 'V', 'K_portmean', 'K_portmean_s', 'K std'))
for p in sorted(pv, key=lambda q: arms['V + K_portmean'][q] - pv[q]):
    print('%-18s %8.3f %8.3f %8.3f %8.3f' % (
        p, pv[p], arms['V + K_portmean'][p], arms['V + K_portmean_shuffled'][p], arms['K(0-70) standardized'][p]))

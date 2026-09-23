"""e57: the control e56 was missing -- same-capacity model, with vs without knowledge.

e56's joint_gbm (HistGradientBoostingClassifier on [z, k]) scored far above the ridge V, but it changed
BOTH the model class and the presence of knowledge, so its gain cannot be attributed. This runs the
missing arm: the identical GBM on z alone, plus the ridge arms, on the same per-port LOO.

Arms: ridge_V | ridge_VK | gbm_z | gbm_zk | gbm_zk_permuted (row-shuffled knowledge, the leakage check)
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
ARMS = ['ridge_V', 'ridge_VK', 'gbm_z', 'gbm_zk', 'gbm_zk_shuf']


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


rng = np.random.default_rng(0)
sh = rng.permutation(len(rows))
Ksh = Kp[sh]

print('%-18s %8s %8s %8s %8s %8s' % tuple(['port'] + ARMS))
print('-' * 72)
out = {a: [] for a in ARMS}
for p in PORT_U:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    a = 1.0; best = -1
    for al in ALPHAS:
        sc = []
        for q in [x for x in PORT_U[:3] if x != p]:
            itr = np.where(np.isin(ports, PORT_U) & (ports != p) & (ports != q))[0]
            ite = np.where(ports == q)[0]
            if len(itr) < 200:
                continue
            Aq, Bq = kstd(itr, ite)
            sc.append(ba(y[ite], RidgeClassifier(alpha=al, class_weight='balanced').fit(
                np.c_[X[itr], Aq], y[itr]).predict(np.c_[X[ite], Bq])))
        if sc and float(np.mean(sc)) > best:
            best, a = float(np.mean(sc)), al
    A, B = kstd(tr, te)
    A_sh, B_sh = kstd(tr, te)          # same shape; use the shuffled table below
    Ksh_tr = Ksh[tr][:, LEGAL]; Ksh_te = Ksh[te][:, LEGAL]
    mu = Ksh_tr.mean(0); s = Ksh_tr.std(0); s[s < 1e-9] = 1.0
    Ksh_tr = (Ksh_tr - mu) / s; Ksh_te = (Ksh_te - mu) / s
    G = dict(max_iter=300, max_depth=4, random_state=0)
    preds = {
        'ridge_V': RidgeClassifier(alpha=a, class_weight='balanced').fit(X[tr], y[tr]).predict(X[te]),
        'ridge_VK': RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B]),
        'gbm_z': HistGradientBoostingClassifier(**G).fit(X[tr], y[tr]).predict(X[te]),
        'gbm_zk': HistGradientBoostingClassifier(**G).fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B]),
        'gbm_zk_shuf': HistGradientBoostingClassifier(**G).fit(np.c_[X[tr], Ksh_tr], y[tr]).predict(np.c_[X[te], Ksh_te]),
    }
    for nm, pr in preds.items():
        out[nm].append(ba(y[te], pr))
    print('%-18s %8.3f %8.3f %8.3f %8.3f %8.3f' % (p, *[out[nm][-1] for nm in ARMS]), flush=True)

print()
rv = np.array(out['ridge_V'])
for nm in ARMS[1:]:
    v = np.array(out[nm]); d = (v - rv) * 100
    print('%-12s mean %.4f  Δ vs ridge_V %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
gzk = np.array(out['gbm_zk']); gz = np.array(out['gbm_z']); gs = np.array(out['gbm_zk_shuf'])
print()
print('关键对照（同容量 GBM）: gbm_zk − gbm_z = %+.2f pp   Wilcoxon p=%.4f'
      % ((gzk - gz).mean() * 100, stats.wilcoxon(gzk, gz).pvalue))
print('                     gbm_zk − gbm_zk_shuf = %+.2f pp   p=%.4f'
      % ((gzk - gs).mean() * 100, stats.wilcoxon(gzk, gs).pvalue))

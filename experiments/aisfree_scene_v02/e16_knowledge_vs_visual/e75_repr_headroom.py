"""e75: representation headroom -- how much is the 128-dim PCA costing us, and does a stronger
visual baseline change the knowledge's gain?

Representations, all fitted PER FOLD on the training sources only (this also fixes the standing caveat
that the shipped visual_projection.npz PCA was fitted globally, i.e. on the target port too):
  pca128   the current input, refit correctly per fold
  pca512 / pca1024 / raw2048
Arms per representation: ridge_V (visual only) and ridge_VK (visual + within-port percentile knowledge).
No GBM here on purpose: the question is the representation axis, so only one variable moves.

Outputs the cross-port LOO mean / positives / worst per representation, plus the knowledge delta under
each -- i.e. whether the 'weaker baseline gains more' law extends to the representation axis.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
RAW = np.load(ROOT / 'pooled_features.npy').astype(np.float32)          # (38091, 2048)
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros((len(y), 71), np.float32)
for pj in PORT_U:
    m = ports == pj
    Kp[np.ix_(m, range(71))] = K[np.ix_(m, LEGAL)].argsort(0).argsort(0) / max(1, int(m.sum()) - 1)

REPS = [('pca128', 128), ('pca512', 512), ('pca1024', 1024), ('raw2048', None)]
SUBS = 20000
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def std_k(tr, te):
    A, B = Kp[tr], Kp[te]
    mu, s = A.mean(0), A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


res = {r: {a: [] for a in ('V', 'VK')} for r, _ in REPS}
print('%-18s %s' % ('port', ' '.join('%16s' % r for r, _ in REPS)), flush=True)
for pi, p in enumerate(PORT_U):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    A, B = std_k(trs, te)
    line = []
    for nm, nc in REPS:
        Xtr_full, Xte_full = RAW[trs], RAW[te]
        if nc is None or nc >= RAW.shape[1]:
            Ztr, Zte = Xtr_full, Xte_full
        else:
            pc = PCA(n_components=nc, svd_solver='randomized', random_state=20260920).fit(Xtr_full)  # sources only
            Ztr, Zte = pc.transform(Xtr_full), pc.transform(Xte_full)
        rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
        rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, A], y[trs])
        pv = rv.predict(Zte); pk = rk.predict(np.c_[Zte, B])
        res[nm]['V'].append(ba(y[te], pv)); res[nm]['VK'].append(ba(y[te], pk))
        line.append('%6.3f→%6.3f' % (res[nm]['V'][-1], res[nm]['VK'][-1]))
    print('%-18s %s' % (p, ' '.join('%16s' % x for x in line)), flush=True)

print()
print('%-10s %9s %9s %9s %9s %8s %8s' % ('rep', 'V mean', 'VK mean', 'Δknowledge', 'worstΔ', 'V正', 'VK正'))
for nm, _ in REPS:
    v = np.array(res[nm]['V']); vk = np.array(res[nm]['VK']); d = (vk - v) * 100
    print('%-10s %9.4f %9.4f %+9.2f %9.2f %8d %8d' % (
        nm, v.mean(), vk.mean(), d.mean(), d.min(), int((v > 0).sum()), int((vk > 0).sum())))
print()
base = np.array(res['pca128']['V'])
for nm, _ in REPS[1:]:
    x = np.array(res[nm]['V']); m = ~(np.isnan(x) | np.isnan(base))
    print('配对 %s − pca128(视觉单臂): %+6.2f pp  p=%.4f' % (nm, (x[m] - base[m]).mean() * 100,
                                                          stats.wilcoxon(x[m], base[m]).pvalue))
np.save('e75_res.npy', np.array([res[nm]['V'] + res[nm]['VK'] for nm, _ in REPS], dtype=object), allow_pickle=True)

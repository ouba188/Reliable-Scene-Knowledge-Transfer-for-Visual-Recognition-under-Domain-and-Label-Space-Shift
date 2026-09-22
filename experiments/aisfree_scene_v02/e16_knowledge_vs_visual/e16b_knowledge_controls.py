"""e16b: is the raw-knowledge +4pp real, or a port-prior / scale shortcut?

Controls (all strict per-port LOO, all deployment-legal):
  V              visual only
  K_raw          V + raw 82-dim knowledge
  K_rowshuf      V + knowledge rows shuffled GLOBALLY      (breaks object<->knowledge pairing)
  K_withinshuf   V + knowledge rows shuffled WITHIN each port (keeps port distribution, breaks pairing)
  K_portmean     V + each object's PORT MEAN knowledge only (a per-port constant, no per-object info)
  K_colshuf      V + knowledge columns permuted            (sanity: a linear model is invariant to this)
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows])
y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
n = len(rows)
from sklearn.linear_model import RidgeClassifier


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


rng = np.random.default_rng(20260922)
gperm = rng.permutation(n)                     # 全局行置换
cperm = rng.permutation(K.shape[1])            # 列置换（应无影响）

# 港内行置换 + 港均值
K_within = K.copy()
K_portmean = K.copy()
for p in set(ports.tolist()):
    idx = np.where(ports == p)[0]
    K_within[idx] = K[rng.permutation(idx)]
    K_portmean[idx] = K[idx].mean(0, keepdims=True)

arms = {a: [] for a in ['V', 'K_raw', 'K_rowshuf', 'K_withinshuf', 'K_portmean', 'K_colshuf']}
featmap = {
    'V': lambda idx: X[idx],
    'K_raw': lambda idx: np.c_[X[idx], K[idx]],
    'K_rowshuf': lambda idx: np.c_[X[idx], K[gperm][idx]],
    'K_withinshuf': lambda idx: np.c_[X[idx], K_within[idx]],
    'K_portmean': lambda idx: np.c_[X[idx], K_portmean[idx]],
    'K_colshuf': lambda idx: np.c_[X[idx], K[idx][:, cperm]],
}
for p in sorted(set(ports.tolist())):
    tr = np.where(ports != p)[0]
    te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    for a, fn in featmap.items():
        clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(fn(tr), y[tr])
        arms[a].append(ba(y[te], clf.predict(fn(te))))

base = float(np.mean(arms['V']))
print('%-14s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 34)
for a in ['V', 'K_raw', 'K_rowshuf', 'K_withinshuf', 'K_portmean', 'K_colshuf']:
    m = float(np.mean(arms[a]))
    print('%-14s %8.4f %+10.4f' % (a, m, m - base))

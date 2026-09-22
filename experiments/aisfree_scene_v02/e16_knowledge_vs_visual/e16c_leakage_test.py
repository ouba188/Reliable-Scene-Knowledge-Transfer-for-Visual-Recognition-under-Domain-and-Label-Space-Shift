"""e16c: leakage test — split the 82 knowledge dims by evidence source.

  0-41   OSM facility proximity/contrast (deployment-legal, static map)
  42-59  OSM chain/freight features   (deployment-legal)
  60-70  detection-derived spatial    (SAR-only, deployment-legal)
  71-81  AIS-derived density/heading  (LEAKY: AIS is the label source)
"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
feats = json.load((ROOT / 'knowledge' / 'relation_features.json').open(encoding='utf-8'))
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1; n = len(rows)
from sklearn.linear_model import RidgeClassifier

GROUPS = {
    'V': [],
    'OSM_facility(0-41)': list(range(0, 42)),
    'OSM_chain(42-59)': list(range(42, 60)),
    'detection(60-70)': list(range(60, 71)),
    'AIS(71-81)': list(range(71, 82)),
    'all_82': list(range(82)),
    'all_minus_AIS': list(range(0, 71)),
}
print('AIS 组维度:', [feats[i] for i in range(71, 82)])
print()


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


rng = np.random.default_rng(20260922)
res = {a: [] for a in GROUPS}
res['AIS_shuffled'] = []
for p in sorted(set(ports.tolist())):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    for a, dims in GROUPS.items():
        ftr = X[tr] if not dims else np.c_[X[tr], K[tr][:, dims]]
        fte = X[te] if not dims else np.c_[X[te], K[te][:, dims]]
        clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(ftr, y[tr])
        res[a].append(ba(y[te], clf.predict(fte)))
    # AIS 行打乱对照
    ais = np.array([71 + i for i in range(11)])
    g = rng.permutation(n)
    ftr = np.c_[X[tr], K[g][tr][:, list(range(71, 82))]]
    fte = np.c_[X[te], K[g][te][:, list(range(71, 82))]]
    clf = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(ftr, y[tr])
    res['AIS_shuffled'].append(ba(y[te], clf.predict(fte)))

base = float(np.mean(res['V']))
print('%-22s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 42)
for a in ['V', 'OSM_facility(0-41)', 'OSM_chain(42-59)', 'detection(60-70)', 'AIS(71-81)',
          'all_82', 'all_minus_AIS', 'AIS_shuffled']:
    m = float(np.mean(res[a]))
    print('%-22s %8.4f %+10.4f' % (a, m, m - base))

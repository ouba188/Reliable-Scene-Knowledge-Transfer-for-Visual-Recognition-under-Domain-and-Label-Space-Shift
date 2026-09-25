"""e129: can the SAR-derived density/heading block replace the AIS-derived one?

The user's point is right and it decides how the violation should be repaired: ship density and heading ARE obtainable at
inference time from the SAR detections themselves, no AIS needed. The knowledge schema says the same thing structurally --
dims 60:71 (detection_*) and 71:82 (ais_*) are the SAME eleven quantities computed from two sources (SAR detections vs the AIS
traffic field). So the question is empirical: does the SAR-derived block carry the same value as the AIS-derived block?

Six arms, identical folds, source ports only: V | V+detection(60:71) | V+AIS(71:82) | V+both(60:82) |
V+legal(0:71) | V+full(0:82). Ridge, per-port balanced accuracy, paired against V.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
ARMS = {
    'V': None,
    'V+detection (SAR ✓ 合规)': (60, 71),
    'V+AIS (✗ 违规)': (71, 82),
    'V+both density blocks': (60, 82),
    'V+legal all (0:71)': (0, 71),
    'V+full (0:82)': (0, 82),
}


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
res = {a: [] for a in ARMS}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 20:
        continue
    for a, sl in ARMS.items():
        Z = X[tr] if sl is None else np.c_[X[tr], K[tr][:, sl[0]:sl[1]]]
        Q = X[te] if sl is None else np.c_[X[te], K[te][:, sl[0]:sl[1]]]
        mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
        m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[tr])
        res[a].append(ba(y[te], m.predict((Q - mu) / sd)))
    print('%-16s done (%d)' % (p, len(res['V'])), flush=True)

b = np.array(res['V'])
print('')
print('%-28s %9s %12s %9s %9s' % ('arm', 'BA', 'ΔBA(pp)', '正港/24', '配对 p'))
for a in ARMS:
    v = np.array(res[a]); d = (v - b) * 100
    pv = stats.wilcoxon(v, b).pvalue if np.any(v != b) else float('nan')
    print('%-28s %9.4f %+12.2f %9d %9.4f' % (a, v.mean(), d.mean(), int((d > 0).sum()), pv))
print('')
det = (np.array(res['V+detection (SAR ✓ 合规)']) - b) * 100
ais = (np.array(res['V+AIS (✗ 违规)']) - b) * 100
print('SAR 检测块 %+.2fpp ｜ AIS 块 %+.2fpp ⇒ %s' % (
    det.mean(), ais.mean(),
    'SAR 块可顶替 AIS 块 ✓✓（违规可合法修复 ✓）' if det.mean() >= 0.7 * ais.mean() else
    'SAR 块顶不上 AIS 块 ✗（AIS 交通场带的是条带外信息 ✓，不可合法替代 ✗）'))
print('两者差异配对 p=%.4f' % stats.wilcoxon(np.array(res['V+detection (SAR ✓ 合规)']), np.array(res['V+AIS (✗ 违规)'])).pvalue)

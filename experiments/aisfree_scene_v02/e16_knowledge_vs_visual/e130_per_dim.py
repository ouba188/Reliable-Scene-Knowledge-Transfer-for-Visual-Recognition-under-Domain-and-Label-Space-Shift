"""e130: which of the eleven parallel density/heading quantities actually carry value?

The user's point (detections can supply density/heading without AIS, and a detection-derived heading is more accurate than an
AIS heading offset in time) is true about COMPUTABILITY but the block-level test (e129) says the SAR-derived block carries
+0.37 pp against the AIS block's +4.23 pp. This splits the block into its eleven quantities and measures each one alone, so
the answer is per-quantity rather than per-block: if the AIS version of a quantity is strong while the detection version is
weak, the value is coming from the AIS field's port-wide coverage, not from the heading accuracy.

Also reports corr(detection version, AIS version) per quantity: high correlation with different gains means the engines agree
on the feature but the AIS one is better resolved; low correlation means they are measuring different things.
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
NAME = [f'd{i}' for i in range(11)]        # 60+i = detection version, 71+i = AIS version
PAIRS = [(i, 60 + i, 71 + i) for i in range(11)]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
bV, res_d, res_a = [], {i: [] for i in range(11)}, {i: [] for i in range(11)}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 20:
        continue
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Zv = (X[tr] - mu) / sd; Qv = (X[te] - mu) / sd
    bV.append(ba(y[te], RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Zv, y[tr]).predict(Qv)))
    for i, cd, ca in PAIRS:
        for store, col in ((res_d, cd), (res_a, ca)):
            ktr = K[tr][:, col][:, None]; kte = K[te][:, col][:, None]
            km, ks = ktr.mean(0), ktr.std(0) + 1e-6
            m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Zv, (ktr - km) / ks], y[tr])
            store[i].append(ba(y[te], m.predict(np.c_[Qv, (kte - km) / ks])))
    print('%-16s done (%d)' % (p, len(bV)), flush=True)

b = np.array(bV)
print('')
print('%-3s %-34s %12s %12s %9s %9s' % ('#', 'quantity (detection_* / ais_*)', 'det ΔBA(pp)', 'ais ΔBA(pp)', 'corr', 'AIS p'))
for i, cd, ca in PAIRS:
    dv = (np.array(res_d[i]) - b) * 100; av = (np.array(res_a[i]) - b) * 100
    pv = stats.wilcoxon(np.array(res_a[i]), b).pvalue if np.any(np.array(res_a[i]) != b) else float('nan')
    cc = float(np.corrcoef(K[:, cd], K[:, ca])[0, 1])
    print('%-3d %-34s %+12.2f %+12.2f %9.3f %9.4f' % (i, 'q%d' % i, dv.mean(), av.mean(), cc, pv))
print('')
dbest = max(range(11), key=lambda i: ((np.array(res_d[i]) - b) * 100).mean())
abest = max(range(11), key=lambda i: ((np.array(res_a[i]) - b) * 100).mean())
print('检测版最强 q%d (%+.2fpp) ｜ AIS 版最强 q%d (%+.2fpp)'
      % (dbest, ((np.array(res_d[dbest]) - b) * 100).mean(), abest, ((np.array(res_a[abest]) - b) * 100).mean()))
print('⇒ %s' % ('AIS 的价值在于"场覆盖"而非单个量精度 ✓（检测版同名量都很弱 ✓）'
                if ((np.array(res_d[dbest]) - b) * 100).mean() < 1.0 else '某些量检测版也能带值 ✓'))

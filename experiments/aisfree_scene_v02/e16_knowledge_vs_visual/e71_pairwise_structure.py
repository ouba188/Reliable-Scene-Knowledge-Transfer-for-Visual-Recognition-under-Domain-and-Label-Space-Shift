"""e71: does PAIRWISE structure exist? (5-minute gate before building any GNN/attention module)

If the harm is only a node property (a class's overall sharing), a scalar gate suffices -- and e70 already
showed adding the scalar to the gate changes nothing. A learnable graph module is only justified if the
SPECIFIC pair (c -> c') is predictable from facility overlap.

Measured, all from frozen artifacts:
  S(c,c')      = cosine of the two classes' mean within-port percentile knowledge profiles (+ a top-dim
                 Jaccard variant as robustness);
  harm flow    = (# of instances whose TRUE class is c, which the visual arm got right and the knowledge
                 arm then pushed to c')   -- the directed damage;
  rescue flow  = the opposite direction.
Test: Spearman over the 56 ordered class pairs, and the AUC of S for separating the pairs that actually
carry harm flow from the pairs that do not.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
NAMES = {0: 'bulk', 1: 'fish', 2: 'gcargo', 3: 'pchem', 4: 'cont', 5: 'crude', 6: 'tug', 7: 'offsh'}

# within-port percentile knowledge (e45 transform), the same coordinate the models use
Kp = np.zeros((len(y), C and len(LEGAL)))
Kp = np.zeros((len(y), len(LEGAL)))
for pj in sorted(set(ports.tolist())):
    m = ports == pj
    Kp[np.ix_(m, range(len(LEGAL)))] = K[np.ix_(m, LEGAL)].argsort(0).argsort(0) / max(1, int(m.sum()) - 1)

prof = np.array([Kp[y == c].mean(0) for c in range(C)])
S = np.zeros((C, C))
for a in range(C):
    for b in range(C):
        if a == b:
            continue
        va, vb = prof[a], prof[b]
        S[a, b] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
top = [set(np.argsort(prof[c])[::-1][:10].tolist()) for c in range(C)]
J = np.zeros((C, C))
for a in range(C):
    for b in range(C):
        J[a, b] = len(top[a] & top[b]) / max(1, len(top[a] | top[b]))

# ---- directed flows from the frozen e61 predictions ----
d61 = pickle.load(open('e61_preds.pkl', 'rb'))     # our own artifact: a list of per-port dumps
harm = np.zeros((C, C)); resc = np.zeros((C, C)); ntrue = np.zeros(C)
for rec in d61:
    yy = rec['y']; v = rec['pred']['ridge_V']; vk = rec['pred']['ridge_VK']
    for t, pv, pk in zip(yy, v, vk):
        ntrue[t] += 1
        if pv == t and pk != t:
            harm[t, pk] += 1
        elif pv != t and pk == t:
            resc[pv, t] += 1
print('真实样本数/类:', {NAMES[c]: int(ntrue[c]) for c in range(C)})
print('误伤流总量 %d  救回流总量 %d' % (harm.sum(), resc.sum()))

pairs = [(a, b) for a in range(C) for b in range(C) if a != b]
s = np.array([S[a, b] for a, b in pairs]); j = np.array([J[a, b] for a, b in pairs])
h = np.array([harm[a, b] for a, b in pairs]); r = np.array([resc[a, b] for a, b in pairs])
# rate-normalise so a big class does not dominate purely by size
hr = h / np.maximum(ntrue[[a for a, b in pairs]], 1)
rr = r / np.maximum(ntrue[[b for a, b in pairs]], 1)
print()
for nm, x in [('cosine S', s), ('Jaccard J', j)]:
    print('%-10s vs 误伤率: Spearman %+.3f (p=%.4f) ｜ vs 救回率: %+.3f (p=%.4f)' % (
        nm, stats.spearmanr(x, hr).statistic, stats.spearmanr(x, hr).pvalue,
        stats.spearmanr(x, rr).statistic, stats.spearmanr(x, rr).pvalue))
print()
print('S 区分"有误伤流的对"vs"无误伤的对": AUC %.3f  (有流的对 %d/%d)' % (
    roc_auc_score((h > 0).astype(int), s), int((h > 0).sum()), len(h)))
print('J 同上: AUC %.3f' % roc_auc_score((h > 0).astype(int), j))
print()
print('误伤最多的 12 个类别对（真实类 → 被推向的类）:')
for idx in np.argsort(-h)[:12]:
    a, b = pairs[idx]
    if h[idx] == 0:
        continue
    print('  %-7s → %-7s  误伤 %4d ｜ 该对 S=%.3f J=%.3f ｜ 该类样本 %4d' % (
        NAMES[a], NAMES[b], int(h[idx]), S[a, b], J[a, b], int(ntrue[a])))

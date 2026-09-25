"""e174 (takeover, step 3): make the fine-level half deployable by using the ONE mechanism that transfers -- rank + budget.

e173 showed the partition method's fine half only works under an oracle coverage rule: asking "is the TRUE class in a singleton
block" beats a matched-coverage random abstention (p=0.0068), while the deployable version of that question collapses
(delta -0.318, p<0.0001). The diagnosis, from the independent order/value law, is that a coverage RULE is a threshold-shaped
object and thresholds do not transfer.

So replace the rule with a budget, exactly as the module does elsewhere: rank the target port's instances by the FINE
classifier's own margin (label-free, no calibration), decide at fine granularity for the top-k%, and compare against a
matched-size random selection -- paired over the 24 ports.

Pre-registered: at each k in {10, 20, 50}%, fine BA on the ranked top-k% must exceed the random k% with paired p < 0.05.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
ART = BASE / 'artifacts/eight_class_adaptive_20260916'
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
KS = (0.10, 0.20, 0.50, 1.00)
NREP = 30
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
rank, rand = {k: [] for k in KS}, {k: [] for k in KS}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 50:
        continue
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[tr] - mu) / sd, y[tr])
    L = m.decision_function((X[te] - mu) / sd)
    if L.shape[1] != C:
        Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
    pred = L.argmax(1)
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]                      # fine-class margin, label-free
    yy = y[te]
    for k in KS:
        n = max(1, int(k * len(te)))
        sel = np.argsort(-margin)[:n]
        rank[k].append(ba(yy[sel], pred[sel]))
        # one permutation, one subset for BOTH labels and predictions -- two independent draws would score it at chance
        rand[k].append(float(np.mean([(lambda idx: ba(yy[idx], pred[idx]))(rng.permutation(len(te))[:n])
                                      for _ in range(NREP)])))
    print('%-16s n=%5d 全量 %.4f' % (p, len(te), ba(yy, pred)), flush=True)

print('')
print('%-8s %14s %14s %10s %10s' % ('budget', 'ranked fine BA', 'random fine BA', 'delta', 'paired p'))
for k in KS:
    a, b = np.array(rank[k]), np.array(rand[k])
    pv = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
    print('%-8s %14.4f %14.4f %+10.4f %10.4f' % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(), pv))
# budget curve: what fraction of the full-pool fine BA does each budget keep?
full = np.array(rank[1.00])
print('')
for k in (0.10, 0.20, 0.50):
    a = np.array(rank[k])
    print('  预算 %2.0f%% 细类 BA %.4f ⇒ 相对全量 %.1f%%' % (100 * k, a.mean(), 100 * a.mean() / full.mean()))
ok = any(np.array(rank[k]).mean() > np.array(rand[k]).mean() and stats.wilcoxon(rank[k], rand[k]).pvalue < 0.05
         for k in (0.10, 0.20, 0.50))
print('')
print('预注册判据（某个 k 上 排序 > 随机 且 p<0.05）: %s' % ('成立 ✓✓ —— 细类半边可用「序+预算」做成可部署 ✓' if ok else '未成立 ✗'))

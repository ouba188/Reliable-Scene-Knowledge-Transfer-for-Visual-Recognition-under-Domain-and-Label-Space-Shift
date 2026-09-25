"""e176: replicate the fine-margin budget rescue on the NEW pool (71,801 chips, 24 ports, 17 classes).

e174 rescued the granularity line's fine half on the old 8-class pool by replacing a coverage RULE (a threshold-shaped object)
with a BUDGET: rank by the classifier's own margin, decide at full granularity only for the top-k%. It needs no knowledge
dimension, so it can be tested straight away on the rebuilt pool -- the paper's headline dataset, with 17 classes instead of 8 and
features from the 64-px field of view.

Two things are measured:
  (a) the same selective-accuracy effect (ranked top-k% vs a matched-size random k%), paired over the 24 ports;
  (b) a by-product that matters for the open-set direction: does the margin ranking also concentrate the KNOWN classes, i.e. is
      the top-k% by margin more known-heavy than a random k%? If so, the same score serves both decisions.

Pre-registered: at k = 10% and 20%, ranked BA must exceed the matched random BA with paired p < 0.05.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
FEAT = BASE / 'features_all' / 'pca512_all.npy'
IDX = BASE / 'dataset_all' / 'index.csv'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
KS = (0.10, 0.20, 0.50, 1.00)
NREP = 30
rng = np.random.default_rng(0)

X = np.load(FEAT).astype(np.float32)
rows = list(csv.DictReader(IDX.open(encoding='utf-8')))
cls = np.array([r['class'] for r in rows]); ports = np.array([r['port'] for r in rows])
names = sorted(set(cls.tolist()))
y = np.array([names.index(c) for c in cls])
C = len(names)
known = np.array([c in KNOWN8 for c in cls])
print('新池 %d 片 ｜ %d 港 ｜ %d 类（已知 %d / 未知 %d）｜ 特征 %s'
      % (len(rows), len(set(ports.tolist())), C, int(known.sum()), int((~known).sum()), X.shape), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
rank, rand, knownrate_r, knownrate_n = {k: [] for k in KS}, {k: [] for k in KS}, {k: [] for k in KS}, {k: [] for k in KS}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 50:
        continue
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[tr] - mu) / sd, y[tr])
    L = m.decision_function((X[te] - mu) / sd)
    if L.shape[1] != C:
        Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    pred = L.argmax(1); yy = y[te]; kn = known[te]
    for k in KS:
        n = max(1, int(k * len(te)))
        sel = np.argsort(-margin)[:n]
        rank[k].append(ba(yy[sel], pred[sel]))
        knownrate_r[k].append(float(kn[sel].mean()))
        picks = rng.permutation(len(te))[:n]
        rand[k].append(ba(yy[picks], pred[picks]))
        knownrate_n[k].append(float(kn[picks].mean()))
    print('%-16s n=%5d 全量 %.4f ｜ 未知占比 %.2f' % (p, len(te), ba(yy, pred), 1 - kn.mean()), flush=True)

print('')
print('%-8s %12s %12s %10s %10s %14s' % ('budget', 'ranked BA', 'random BA', 'delta', 'paired p', 'known rate r/n'))
for k in KS:
    a, b = np.array(rank[k]), np.array(rand[k])
    pv = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
    print('%-8s %12.4f %12.4f %+10.4f %10.4f %14s'
          % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(), pv,
             '%.2f / %.2f' % (np.mean(knownrate_r[k]), np.mean(knownrate_n[k]))))
full = np.array(rank[1.00])
print('')
for k in (0.10, 0.20, 0.50):
    a = np.array(rank[k])
    print('  预算 %2.0f%% 秩选 BA %.4f ⇒ 全量的 %.1f%%' % (100 * k, a.mean(), 100 * a.mean() / full.mean()))
ok = any(np.array(rank[k]).mean() > np.array(rand[k]).mean() and stats.wilcoxon(rank[k], rand[k]).pvalue < 0.05
         for k in (0.10, 0.20))
print('')
print('预注册判据（k=10/20% 上 排序>随机 且 p<0.05）: %s' % ('成立 ✓✓ —— 新池复现 ✓' if ok else '未成立 ✗'))
kr = np.mean([np.mean(knownrate_r[k]) for k in (0.10, 0.20)])
kn_ = np.mean([np.mean(knownrate_n[k]) for k in (0.10, 0.20)])
print('已知类占比：秩选 %.3f vs 随机 %.3f ⇒ %s' % (kr, kn_, 'margin 同时偏好已知类 ✓（一个分数两用 ✓）' if kr > kn_ + 0.02 else '不偏好已知类 ⇒ 两用不成立 ✗'))

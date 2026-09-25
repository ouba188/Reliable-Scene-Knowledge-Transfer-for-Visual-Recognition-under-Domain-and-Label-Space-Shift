"""e138: make the COMPLIANT expert stronger -- the last un-refuted lever.

Everything else was measured dead (sequence, source count, conformal, SAR field). The compliant expert (knowledge dims 0:71) is
the current module's weak point: only +2.16 pp over visual, p=0.0604, which caps the module's absolute value.

The compliant block has three documented parts (e15's layout): facility 0:42, chain 42:60, detection 60:71. Rather than an
open-ended search, test all seven non-empty subsets with two classifiers (ridge, GBM) -- 14 arms, no tuning beyond that.

Pre-registered: some legal configuration must reach DeltaBA >= +3.00 pp with paired p < 0.05 (the baseline legal expert is
+2.16 pp at p=0.0604, so this asks for ~40% more gain).
"""
import csv
import itertools
import os
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
KALL = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]   # compliant only
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
PARTS = {'facility': (0, 42), 'chain': (42, 60), 'detection': (60, 71)}
SUBSETS = []
for r in range(1, 4):
    for combo in itertools.combinations(PARTS, r):
        cuts = [PARTS[c] for c in combo]
        SUBSETS.append(('+'.join(c[:4] for c in combo), cuts))
PU = sorted(set(ports.tolist()))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


def cols(cuts):
    return np.concatenate([np.arange(a, b) for a, b in cuts])


base = []
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 20:
        continue
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[tr] - mu) / sd, y[tr])
    base.append(ba(y[te], m.predict((X[te] - mu) / sd)))
b = np.array(base)
print('视觉基线 BA %.4f（%d 港）' % (b.mean(), len(b)), flush=True)

print('')
print('%-14s %-7s %10s %10s %9s %9s' % ('subset', 'clf', 'BA', 'ΔBA(pp)', '正港', '配对 p'))
best = None
for name, cuts in SUBSETS:
    K = KALL[:, cols(cuts)]
    for clf in ('ridge', 'gbm'):
        acc = []
        for p in PU:
            tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
            if len(tr) < 500 or len(te) < 20:
                continue
            Z = np.c_[X[tr], K[tr]]; Q = np.c_[X[te], K[te]]
            mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
            if clf == 'ridge':
                m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[tr])
                pred = m.predict((Q - mu) / sd)
            else:
                m = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit((Z - mu) / sd, y[tr])
                pred = m.predict((Q - mu) / sd)
            acc.append(ba(y[te], pred))
        a = np.array(acc) if len(acc) == len(b) else np.array(acc[:len(b)])
        d = (a - b) * 100
        pv = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
        print('%-14s %-7s %10.4f %+10.2f %9d %9.4f' % (name, clf, a.mean(), d.mean(), int((d > 0).sum()), pv), flush=True)
        if best is None or d.mean() > best[0]:
            best = (d.mean(), pv, name, clf, len(cols(cuts)))
print('')
print('最佳合规配置: %s + %s ｜ ΔBA %+.2fpp ｜ p=%.4f ｜ 维度 %d'
      % (best[2], best[3], best[0], best[1], best[4]))
print('预注册判据（某合规配置 ΔBA >= +3.00pp 且 p<0.05）: %s'
      % ('成立 ✓✓ —— 合规专家可做实 ✓' if (best[0] >= 3.0 and best[1] < 0.05) else '未成立 ✗'))

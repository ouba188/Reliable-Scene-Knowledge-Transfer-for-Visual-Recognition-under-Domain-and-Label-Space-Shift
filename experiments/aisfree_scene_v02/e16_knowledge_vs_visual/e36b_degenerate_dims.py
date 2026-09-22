"""e36b: how many of the 71 'legal' knowledge dims are degenerate (no information at all)?"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
Kraw = k['knowledge'].astype(np.float64); Ksup = k['support']
K = np.where(Ksup, Kraw, 0.0)
feats = json.load((ROOT / 'knowledge' / 'relation_features.json').open(encoding='utf-8'))
LEGAL = list(range(0, 71))

dead, dead_names = [], []
low, low_names = [], []
for i in LEGAL:
    x = K[:, i]
    supp = Ksup[:, i].mean() * 100
    sd = float(x.std())
    nz = float((x != 0).mean() * 100)
    if sd < 1e-9:
        dead.append((i, feats[i], supp, nz)); dead_names.append(feats[i])
    elif sd < 0.01:
        low.append((i, feats[i], sd, supp, nz))

print('=== 完全退化（标准差=0，全常数） : %d / %d 维 ===' % (len(dead), len(LEGAL)))
for i, n, s, z in dead:
    print('  %-3d %-46s support %.1f%%  非零 %.1f%%' % (i, n[:46], s, z))
print()
print('=== 近退化（标准差<0.01）: %d 维 ===' % len(low))
for i, n, sd, s, z in low:
    print('  %-3d %-46s sd %.5f  support %.1f%%' % (i, n[:46], sd, s))
print()
blk = lambda i: 'OSM设施(0-41)' if i < 42 else ('OSM货链(42-59)' if i < 60 else '检测块(60-70)')
from collections import Counter
c = Counter(blk(i) for i, _, _, _ in dead)
print('退化维按块分布:', dict(c))
c2 = Counter(blk(i) for i in LEGAL)
print('各块总维数:', dict(c2))
print()
print('唯一非零值个数中位（全 71 维）: %.0f' % np.median([len(np.unique(K[:, i])) for i in LEGAL]))
print('非零值 >=2 的维度数: %d / %d' % (sum(1 for i in LEGAL if len(np.unique(K[:, i])) >= 2), len(LEGAL)))

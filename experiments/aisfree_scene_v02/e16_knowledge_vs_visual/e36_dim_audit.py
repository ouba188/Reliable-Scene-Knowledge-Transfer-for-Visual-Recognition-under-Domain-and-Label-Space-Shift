"""e36: per-dimension audit of the 71 deployment-legal knowledge dims.

Two questions per dim:
  I(dim; class)  -- how class-informative is it?  (high is fine; but see the next one)
  I(dim; port)   -- how much is it a PORT IDENTITY?  (high => shortcut risk: a port fingerprint can
                    inflate cross-port numbers the way the AIS dims did)
A dim that is strongly port-identifying is flagged, because 'legal' != 'safe': the detection-derived
block (60-70) is SAR-only and legal, yet could still encode "this port has many tugs".

MI via a 16-bin histogram; normalised by H(y). All label-free-tolerable: this is a data audit.
"""
import csv, json
from pathlib import Path

import numpy as np

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
feats = json.load((ROOT / 'knowledge' / 'relation_features.json').open(encoding='utf-8'))
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
PORT_U = sorted(set(ports.tolist()))
pid = np.array([PORT_U.index(p) for p in ports])
LEGAL = list(range(0, 71))
NB = 16


def mi(x, lab, nlab):
    ok = np.isfinite(x)
    x, lab = x[ok], lab[ok]
    if x.std() < 1e-12:
        return 0.0
    edges = np.quantile(x, np.linspace(0, 1, NB + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    j = np.zeros((len(edges) - 1, nlab))
    for bi, li in zip(b, lab):
        j[bi, li] += 1
    p = j / j.sum()
    px = p.sum(1, keepdims=True); pl = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ pl)[nz])).sum())


hy = -sum((np.bincount(y) / len(y))[np.bincount(y) > 0] * np.log((np.bincount(y) / len(y))[np.bincount(y) > 0]))
hp = -sum((np.bincount(pid) / len(pid))[np.bincount(pid) > 0] * np.log((np.bincount(pid) / len(pid))[np.bincount(pid) > 0]))

out = []
for i in LEGAL:
    x = K[:, i]
    out.append((i, feats[i], mi(x, y, len(np.unique(y))) / hy, mi(x, pid, len(PORT_U)) / hp))
out.sort(key=lambda t: -t[3])

print('按"港口身份性"降序（越高越像港口指纹 → shortcut 风险）')
print('%-4s %-42s %8s %8s' % ('idx', 'feature', 'I/class', 'I/port'))
for i, n, a, b in out[:20]:
    print('%-4d %-42s %8.3f %8.3f' % (i, n[:42], a, b))
print('...')
print('%-4s %-42s %8s %8s' % ('idx', 'feature', 'I/class', 'I/port'))
for i, n, a, b in out[-8:]:
    print('%-4d %-42s %8.3f %8.3f' % (i, n[:42], a, b))

arr = np.array([[t[2], t[3]] for t in out])
print()
print('全体: 平均 I/class %.3f, 平均 I/port %.3f' % (arr[:, 0].mean(), arr[:, 1].mean()))
hi = [t for t in out if t[3] > 0.5]
print('高港口身份性 (I/port>0.5) 的维度数: %d' % len(hi))
for i, n, a, b in hi:
    print('   %-3d %-44s I/class %.3f  I/port %.3f%s' % (i, n[:44], a, b, '  <- 检测块' if 60 <= i <= 70 else ''))

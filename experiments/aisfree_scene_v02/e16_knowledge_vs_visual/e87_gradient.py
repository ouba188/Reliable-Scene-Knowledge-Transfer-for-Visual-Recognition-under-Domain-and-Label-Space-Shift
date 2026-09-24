"""e87: the semantic-gradient table with PORT-LOCAL facilities.

e85 could not discriminate because the region-level facility layer put every class at the same ~2.3 km from any
liquid facility. With the port-local extraction (e86) the within-port density is an order of magnitude higher, so
the per-class liquid/dry adjacency becomes meaningful in the direction the mechanism predicts: the lpg/lng tanker
sits at liquid berths, the dredger does not.

Chips' geography comes from geo/<port>.geojson (real lon/lat per product|pol|det), which e85 established is in
the same frame as the port-local facilities (min distance 7 m).
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
GEOJ = Path(r'E:/临时会话/knowledge_set_841/geo')
FP = ROOT / 'facilities_port'
PORTS = ['Antwerp-Bruges', 'Jebel Ali', 'Los Angeles']
LIQUID = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY = ('silo', 'conveyor', 'crane')
R_IN = 10000.0     # in-port radius, metres (10 km: the object table port assignment is regional)

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
geo = {}
for pj in PORTS:
    f = GEOJ / ('%s.geojson' % pj)
    if not f.is_file():
        continue
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        pr = ft['properties']
        c = (ft.get('geometry') or {}).get('coordinates')
        if not c:
            continue
        try:
            ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
            geo[(pj, pr.get('product'), pr.get('pol'), pr.get('det'))] = (
                sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))
        except Exception:
            pass
print('geo 经纬度条目', len(geo))

res = defaultdict(list)
for pj in PORTS:
    f = FP / ('%s.geojson' % pj.replace(' ', '_'))
    if not f.is_file():
        f = FP / ('%s.geojson' % pj)
    if not f.is_file():
        print('缺港区设施', pj)
        continue
    fc = json.load(f.open(encoding='utf-8'))
    L, D, allp = [], [], []
    for ft in fc['features']:
        k = ft['properties']['kind']
        lo, la = ft['geometry']['coordinates']
        allp.append((lo, la))
        if any(x in k for x in LIQUID):
            L.append((lo, la))
        elif any(x in k for x in DRY):
            D.append((lo, la))
    if not allp:
        continue
    lat0 = float(np.mean([p[1] for p in allp]))
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    TALL = cKDTree(np.array([[p[0] * kx, p[1] * ky] for p in allp]))
    TL = cKDTree(np.array([[p[0] * kx, p[1] * ky] for p in L])) if L else None
    TD = cKDTree(np.array([[p[0] * kx, p[1] * ky] for p in D])) if D else None
    n_in = 0
    for r in idx:
        if r['port'] != pj or r['pol'] != 'VV':
            continue
        v = geo.get((pj, r['product'], r['pol'], r['det']))
        if v is None:
            continue
        q = [v[0] * kx, v[1] * ky]
        d_all = TALL.query(q)[0]
        if d_all > R_IN:
            continue
        n_in += 1
        dl = TL.query(q)[0] if TL is not None else np.nan
        dd = TD.query(q)[0] if TD is not None else np.nan
        res[r['class']].append((dl, dd, d_all))
    print('%-16s 设施 %d(液%d/干%d)  港内芯片 %d' % (pj, len(allp), len(L), len(D), n_in))

print('')
print('%-24s %7s %11s %11s %11s %9s' % ('class', 'n(港内)', '液近中位', '干近中位', '到设施中位', '液/干'))
rows = []
for c, v in res.items():
    if len(v) < 15:
        continue
    L1 = np.array([x[0] for x in v]); D1 = np.array([x[1] for x in v]); A1 = np.array([x[2] for x in v])
    rows.append((c, len(v), np.nanmedian(L1), np.nanmedian(D1), np.median(A1),
                 np.nanmedian(L1) / max(1e-6, np.nanmedian(D1))))
for c, n, l, d, a, r_ in sorted(rows, key=lambda z: z[2]):
    print('%-24s %7d %11.0f %11.0f %11.0f %9.2f' % (c, n, l, d, a, r_))
np.save(ROOT / 'e87_gradient.npy', np.array(rows, dtype=object), allow_pickle=True)

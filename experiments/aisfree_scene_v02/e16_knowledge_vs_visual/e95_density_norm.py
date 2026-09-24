"""e95: density-normalised facility proximity -- closing the base-rate alternative from E17w.

E17w found every class closer to a liquid facility than a dry one, which is what a 10:1 liquid:dry facility count
produces whether or not the mechanism is real. The standard correction is to divide each distance by what a
Poisson process of the SAME local density would give (expected nearest-neighbour distance = 1/(2*sqrt(lambda))),
so the score becomes 'how unusual is this vessel's distance, given how many such facilities are around'. If the
per-class ordering then separates in the predicted direction (lpg/crude first), the base-rate was the artifact
and the mechanism survives; if it still does not separate, the physical version of the mechanism is dead.

Reuses e93's calibrated distances (features_244/facility_dist_244.npy) plus the port-local facility sets.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
FP = ROOT / 'facilities_port'
DS = ROOT / 'dataset244'
OUT = ROOT / 'features_244'
LIQUID = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY = ('silo', 'conveyor')
R_LOCAL = 5000.0        # window for the local density estimate

D = np.load(OUT / 'facility_dist_244.npy')
idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
proxy = np.nanmin(D, axis=1)
inport = np.isfinite(proxy) & (proxy <= 3000.0)
print('港内芯片 %d' % int(inport.sum()))

# the calibrated chip positions were not stored, so recompute them the same way e93 did (cheap, same code path)
import gzip
OBJ = KS / 'objects/objects_classed.csv.gz'
want = set((r['product'], r['pol'], r['det']) for r in idx)
xy = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        prod = row['object_id'].split('|')[0]
        k = (prod, row.get('polarization'), row.get('detection_id'))
        if k in want:
            try:
                xy[k] = (float(row['world_x']), float(row['world_y']))
            except (TypeError, ValueError):
                pass
CENTER = {'Antwerp-Bruges': (4.350, 51.270), 'Hamburg': (9.930, 53.540), 'Fujairah': (56.350, 25.160),
          'Jebel Ali': (55.060, 25.010), 'Houston': (-95.270, 29.730), 'Busan': (129.040, 35.100),
          'Callao': (-77.150, -12.050), 'Los Angeles': (-118.270, 33.740)}
span = {}
for k, (x, y) in xy.items():
    pass
for r in idx:
    k = (r['product'], r['pol'], r['det'])
    if k not in xy:
        continue
    pj = r['port']
    s = span.setdefault(pj, [xy[k][0], xy[k][0], xy[k][1], xy[k][1]])
    s[0] = min(s[0], xy[k][0]); s[1] = max(s[1], xy[k][0])
    s[2] = min(s[2], xy[k][1]); s[3] = max(s[3], xy[k][1])
bbox = {}
for r in csv.DictReader((KS / 'ports/port_knowledge.csv').open(encoding='utf-8-sig')):
    try:
        bbox[r['port']] = (float(r['bbox_lon_min']), float(r['bbox_lon_max']),
                           float(r['bbox_lat_min']), float(r['bbox_lat_max']))
    except Exception:
        pass
med = defaultdict(list)
for r in idx:
    k = (r['product'], r['pol'], r['det'])
    if k in xy:
        med[r['port']].append(xy[k])
cal = {}
for pj, (x0, x1, y0, y1) in span.items():
    if pj not in bbox:
        continue
    lon0, lon1, lat0, lat1 = bbox[pj]
    sx = (lon1 - lon0) / max(1e-9, x1 - x0); sy = (lat1 - lat0) / max(1e-9, y1 - y0)
    mx = float(np.median([v[0] for v in med[pj]])); my = float(np.median([v[1] for v in med[pj]]))
    cx, cy = CENTER[pj]
    cal[pj] = (sx, sy, cx - mx * sx, cy - my * sy)

rows = []
for i, r in enumerate(idx):
    if not inport[i]:
        continue
    k = (r['product'], r['pol'], r['det'])
    if k not in xy or r['port'] not in cal:
        continue
    f = FP / ('%s.geojson' % r['port'].replace(' ', '_'))
    if not f.is_file():
        continue
    fc = json.load(f.open(encoding='utf-8'))
    L, Dd, A = [], [], []
    for ft in fc['features']:
        kk = ft['properties']['kind']; lo, la = ft['geometry']['coordinates']
        A.append((lo, la))
        if any(x in kk for x in LIQUID):
            L.append((lo, la))
        elif any(x in kk for x in DRY):
            Dd.append((lo, la))
    if not A:
        continue
    lat0 = float(np.mean([p[1] for p in A]))
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    sx, sy, bx, by = cal[r['port']]
    lon = xy[k][0] * sx + bx; lat = xy[k][1] * sy + by
    q = [lon * kx, lat * ky]
    TA = cKDTree(np.array(A) * [kx, ky])
    # local facility counts within R_LOCAL, then the Poisson expectation for that density
    n_all = len(TA.query_ball_point(q, R_LOCAL))
    area = math.pi * R_LOCAL ** 2
    lam_all = n_all / max(1e-9, area)
    exp_nn = 1.0 / (2.0 * math.sqrt(max(1e-12, lam_all)))
    dl = cKDTree(np.array(L) * [kx, ky]).query(q)[0] if L else np.nan
    dd = cKDTree(np.array(Dd) * [kx, ky]).query(q)[0] if Dd else np.nan
    rows.append((r['class'], dl / exp_nn if np.isfinite(dl) else np.nan,
                 dd / exp_nn if np.isfinite(dd) else np.nan, exp_nn))
print('可算归一化邻近度的芯片 %d' % len(rows))
byc = defaultdict(list)
for c, l, d, e in rows:
    byc[c].append((l, d))
print('')
print('%-24s %6s %12s %12s %9s' % ('class', 'n', '归一液近', '归一干近', '液/干'))
tab = []
for c, v in byc.items():
    if len(v) < 10:
        continue
    L1 = np.array([x[0] for x in v]); D1 = np.array([x[1] for x in v])
    ml, md = float(np.nanmedian(L1)), float(np.nanmedian(D1))
    tab.append((c, len(v), ml, md, ml / max(1e-9, md)))
for c, n, ml, md, r in sorted(tab, key=lambda z: z[2]):
    print('%-24s %6d %12.2f %12.2f %9.2f%s' % (c, n, ml, md, r, ' ★' if r < 1 else ''))
print('')
print('机制预测：lpg_lng_tanker / crude_oil_tanker 的"归一液近"应最小（最异常地贴近液货设施）')
order = [t[0] for t in sorted(tab, key=lambda z: z[2])]
for want_c in ('lpg_lng_tanker', 'crude_oil_tanker', 'dredger', 'bulk_carrier'):
    if want_c in order:
        print('  %-20s 排名 %d/%d' % (want_c, order.index(want_c) + 1, len(order)))

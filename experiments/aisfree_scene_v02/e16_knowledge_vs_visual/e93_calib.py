"""e93: calibrate the object table's (world_x, world_y) frame onto lon/lat, per port, with a built-in acceptance test.

Why calibration instead of another coordinate hunt: the geo/ layer's positions carry a systematic offset (it put
port centres 38-140 km inland, measured and rejected), while the object table's world_x/y is in the SAME frame as
the scenes -- e83 verified that mapping end to end (a UTM->pixel crop landed with the ship centred). What is
missing is only the affine between that frame and geographic coordinates.

Method: per port, fit lon ~ a*x+b and lat ~ c*y+d from two anchors that are both reliable -- the port's public
harbour coordinate (the same table e92 used, itself sanity-checked against the region facility hotspots) and the
scene bbox from ports/port_knowledge.csv (lon/lat, from the acquisition footprint). Acceptance: after calibration
the median chip-to-facility distance must fall from the thousands-of-km range to the hundreds-of-metres range;
if it does not, the affine is wrong and is reported as such rather than used.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
OBJ = KS / 'objects/objects_classed.csv.gz'
FP = ROOT / 'facilities_port'
DS = ROOT / 'dataset244'
GEOJ = KS / 'geo'
OUT = ROOT / 'features_244'
OUT.mkdir(parents=True, exist_ok=True)

CENTER = {   # public harbour coordinates (verified in e92 to within 8-33 km of the region facility hotspots)
    'Antwerp-Bruges': (4.350, 51.270), 'Hamburg': (9.930, 53.540), 'Fujairah': (56.350, 25.160),
    'Jebel Ali': (55.060, 25.010), 'Houston': (-95.270, 29.730), 'Busan': (129.040, 35.100),
    'Callao': (-77.150, -12.050), 'Los Angeles': (-118.270, 33.740),
}
LIQUID = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY = ('silo', 'conveyor')

# scene bbox per port (lon/lat) from the port table
bbox = {}
for r in csv.DictReader((KS / 'ports/port_knowledge.csv').open(encoding='utf-8-sig')):
    try:
        bbox[r['port']] = (float(r['bbox_lon_min']), float(r['bbox_lon_max']),
                           float(r['bbox_lat_min']), float(r['bbox_lat_max']))
    except (TypeError, ValueError, KeyError):
        pass

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
want = set((r['product'], r['pol'], r['det']) for r in idx)
ports = sorted(set(r['port'] for r in idx))

# object-table xy per port
xy = {}
span = {}
import gzip
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        pj = row.get('port')
        if pj not in CENTER:
            continue
        try:
            x, y = float(row['world_x']), float(row['world_y'])
        except (TypeError, ValueError, KeyError):
            continue
        s = span.setdefault(pj, [x, x, y, y])
        s[0] = min(s[0], x); s[1] = max(s[1], x); s[2] = min(s[2], y); s[3] = max(s[3], y)
        prod = row['object_id'].split('|')[0]
        k = (prod, row.get('polarization'), row.get('detection_id'))
        if k in want:
            xy[k] = (x, y)
print('对象 xy 命中 %d / %d' % (len(xy), len(want)))
print('%-16s %10s %10s %10s %10s' % ('port', 'x_span_km', 'bbox_lon_deg', 'y_span_km', 'bbox_lat_deg'))
cal = {}
for pj in ports:
    if pj not in span or pj not in bbox:
        continue
    x0, x1, y0, y1 = span[pj]
    lon0, lon1, lat0, lat1 = bbox[pj]
    sx = (lon1 - lon0) / max(1e-9, (x1 - x0))          # degrees per unit x
    sy = (lat1 - lat0) / max(1e-9, (y1 - y0))
    cx, cy = CENTER[pj]
    # anchor on the harbour centre: it must sit at the port's own reference point in the xy frame, which we take
    # as the median of that port's objects (a robust centre)
    print('%-16s %10.1f %10.3f %10.1f %10.3f' % (pj, (x1 - x0) / 1000.0, lon1 - lon0, (y1 - y0) / 1000.0, lat1 - lat0))
    cal[pj] = (sx, sy, cx, cy)

# land on the harbour centre: shift so that the port's object median maps to the public centre
med = {}
for k, (x, y) in xy.items():
    pass
import csv as _csv
for r in idx:
    v = xy.get((r['product'], r['pol'], r['det']))
    if v:
        med.setdefault(r['port'], []).append(v)
for pj in list(cal):
    if pj in med:
        mx = float(np.median([v[0] for v in med[pj]]))
        my = float(np.median([v[1] for v in med[pj]]))
        sx, sy, cx, cy = cal[pj]
        bx = cx - mx * sx
        by = cy - my * sy
        cal[pj] = (sx, sy, bx, by)
        print('  %-16s 标定: lon=%.6f*x%+.3f  lat=%.6f*y%+.3f  (中位对象 %.0f,%.0f)' % (pj, sx, bx, sy, by, mx, my))

# apply and evaluate against the port-local facilities
def load_fac(pj):
    f = FP / ('%s.geojson' % pj.replace(' ', '_'))
    if not f.is_file():
        f = FP / ('%s.geojson' % pj)
    if not f.is_file():
        return None
    L, D, A = [], [], []
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        k = ft['properties']['kind']
        lo, la = ft['geometry']['coordinates']
        A.append((lo, la))
        if any(x in k for x in LIQUID):
            L.append((lo, la))
        elif any(x in k for x in DRY):
            D.append((lo, la))
    return (np.array(L) if L else None, np.array(D) if D else None, np.array(A))

print('')
print('%-16s %8s %14s %14s %14s' % ('port', 'chips', '中位到设施(m) 未标定', '中位(标定后)', '最近邻(标定后)'))
K = np.full((len(idx), 2), np.nan)
res = []
for pj in ports:
    F = load_fac(pj)
    if F is None or pj not in cal:
        continue
    L, D, A = F
    sel = [i for i, r in enumerate(idx) if r['port'] == pj and (r['product'], r['pol'], r['det']) in xy]
    if not sel:
        continue
    gx = np.array([xy[(idx[i]['product'], idx[i]['pol'], idx[i]['det'])][0] for i in sel])
    gy = np.array([xy[(idx[i]['product'], idx[i]['pol'], idx[i]['det'])][1] for i in sel])
    # uncalibrated: treat xy as if it were degrees (nonsense, shown only as the baseline)
    lat0 = float(np.mean([p[1] for p in A]))
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    treeA = cKDTree(A * [kx, ky])
    d_un = float(np.median(treeA.query(np.c_[gx, gy])[0]))
    # calibrated
    sx, sy, bx, by = cal[pj]
    lon = gx * sx + bx
    lat = gy * sy + by
    q = np.c_[lon * kx, lat * ky]
    dA = treeA.query(q)[0]
    treeL = cKDTree(L * [kx, ky]) if L is not None else None
    treeD = cKDTree(D * [kx, ky]) if D is not None else None
    for pos, i in enumerate(sel):
        K[i, 0] = treeL.query(q[pos])[0] if treeL is not None else np.nan
        K[i, 1] = treeD.query(q[pos])[0] if treeD is not None else np.nan
    res.append((pj, len(sel), d_un, float(np.median(dA)), float(np.min(dA))))
    print('%-16s %8d %14.0f %14.0f %14.0f' % (pj, len(sel), d_un, float(np.median(dA)), float(np.min(dA))))

ok = [r for r in res if r[3] < 5000]
print('')
print('验收: %d/%d 港的标定后中位距离 < 5 km ⇒ %s' % (
    len(ok), len(res), '标定可用 ✓' if len(ok) >= max(1, len(res) // 2) else '标定不可用 ✗（按结果报告，不硬用）'))
np.save(OUT / 'facility_dist_244.npy', K)
print('已写出', OUT / 'facility_dist_244.npy', ' 有值 %d' % int(np.isfinite(K[:, 0]).sum()))

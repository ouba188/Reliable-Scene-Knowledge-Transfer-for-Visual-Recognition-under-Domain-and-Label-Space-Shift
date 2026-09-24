"""e85: NAR knowledge half -- does facility adjacency preferentially absorb the semantically adjacent unknown?

Visual-only absorption is uniform (E17q), which is what a facility-blind model should do. The knowledge is
facility-aware, so the claim to test is: a novel type sitting where a known family's facilities sit gets pulled
INTO that family. Concretely, an lpg tanker at a liquid berth should drift toward crude/product-chemical
tankers, while a dredger should be indifferent.

Per chip: the UTM comes from the object table (world_x/world_y, joined on product|pol|det), converted to lon/lat
via the scene's own CRS, then the distance to the nearest LIQUID facility (storage_tank / pipeline) and to the
nearest DRY facility (silo / conveyor). The visual model's predicted KNOWN class then tells whether the chip
drifts into the liquid family (crude, product_chemical) as liquid adjacency rises.

Ports with local facility layers that also have built chips: Antwerp-Bruges, Jebel Ali, Los Angeles.
"""
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from scipy.spatial import cKDTree
from scipy.special import softmax
from scipy import stats
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
GEOF = Path(r'E:/临时会话/knowledge_set_841/facilities')
OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
SCENES = Path(r'F:/SAR_0922')
FV = ROOT / 'features_vv'
PORTS = ['Antwerp-Bruges', 'Jebel Ali', 'Los Angeles']
LIQUID_KINDS = ('storage_tank', 'pipeline', 'oil', 'tank')
DRY_KINDS = ('silo', 'conveyor')
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ_FAMILY = {'crude_oil_tanker', 'product_chemical_tanker'}

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))
       if r['port'] in PORTS and r['pol'] == 'VV' and r['product']]
print('三港 VV chips', len(idx))
print('按类:', dict(sorted(defaultdict(int, {c: sum(1 for r in idx if r['class'] == c)
                                            for c in set(x['class'] for x in idx)}).items(), key=lambda kv: -kv[1])))

# the object table carries the UTM per (product, pol, det)
want = set((r['product'], r['pol'], r['det']) for r in idx)
xy = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        prod = row['object_id'].split('|')[0]
        k = (prod, row.get('polarization'), row.get('detection_id'))
        if k in want and k not in xy:
            try:
                xy[k] = (float(row['world_x']), float(row['world_y']))
            except (TypeError, ValueError):
                pass
print('UTM 命中 %d/%d' % (len(xy), len(want)))

# convert UTM -> lon/lat using each scene's own CRS
# simpler and exact enough: build the transform per scene CRS once
_xform = {}
_zone = {}


def _mk(lon, lat):
    z = int((lon + 180.0) / 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + z
    src = osr.SpatialReference(); src.ImportFromEPSG(epsg)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference(); dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(src, dst)


# the F: scenes carry a geotransform but no CRS, so the zone is taken from the port's own facility centroid
def lonlat(prod, pol, x, y, pj):
    if pj not in _zone:
        _zone[pj] = None
    t = _zone[pj]
    if t is None:
        return (np.nan, np.nan)
    lo, la, _ = t.TransformPoint(x, y)
    return (lo, la)

# facilities per port
tree = {}
for pj in PORTS:
    f = GEOF / ('%s.geojson' % pj)
    if not f.is_file():
        print('缺设施层', pj); continue
    d = json.load(f.open(encoding='utf-8'))
    liq, dry = [], []
    for ft in d['features']:
        pr = ft['properties']
        kind = (pr.get('facility_kind') or pr.get('man_made') or '').lower()
        c = (ft.get('geometry') or {}).get('coordinates')
        if not c:
            continue
        try:
            ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
            lon = sum(p[0] for p in ring) / len(ring); lat = sum(p[1] for p in ring) / len(ring)
        except Exception:
            continue
        if any(k in kind for k in LIQUID_KINDS):
            liq.append((lon, lat))
        elif any(k in kind for k in DRY_KINDS):
            dry.append((lon, lat))
    lat0 = np.mean([c[1] for c in liq + dry]) if (liq or dry) else 0
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    def mk(pts):
        if not pts:
            return None
        P = np.array([[p[0] * kx, p[1] * ky] for p in pts])
        return cKDTree(P)
    tree[pj] = (mk(liq), mk(dry), kx, ky)
    print('  %-16s liquid %5d  dry %5d' % (pj, len(liq), len(dry)))

for pj, (tl, td, kx, ky) in tree.items():
    # recover the port centroid from the raw facility geojson
    f = GEOF / ('%s.geojson' % pj)
    d = json.load(f.open(encoding='utf-8'))
    lons, lats = [], []
    for ft in d['features'][:400]:
        c = (ft.get('geometry') or {}).get('coordinates')
        if not c:
            continue
        try:
            ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
            lons.append(sum(p[0] for p in ring) / len(ring)); lats.append(sum(p[1] for p in ring) / len(ring))
        except Exception:
            continue
    if lons:
        _zone[pj] = _mk(float(np.mean(lons)), float(np.mean(lats)))
        print('  %-16s UTM zone from centroid (%.2f, %.2f)' % (pj, np.mean(lons), np.mean(lats)))

# the scene/object UTM frame is globally shifted (provenance shows the facilities are true lon/lat), so the
# chips' geography is taken from geo/<port>.geojson which carries a REAL lon/lat polygon per (product, pol, det)
GEOJ = Path(r'E:/临时会话/knowledge_set_841/geo')
_lonlat = {}
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
            lon = sum(p[0] for p in ring) / len(ring); lat = sum(p[1] for p in ring) / len(ring)
        except Exception:
            continue
        _lonlat[(pj, pr.get('product'), pr.get('pol'), pr.get('det'))] = (lon, lat)
print('geo 经纬度条目 %d' % len(_lonlat))

geo = {}
miss = 0
for r in idx:
    v = _lonlat.get((r['port'], r['product'], r['pol'], r['det']))
    if v is None:
        miss += 1
        continue
    geo.setdefault(r['port'], []).append((r, v[0], v[1]))
print('芯片地理命中 %d / 未命中 %d' % (sum(len(v) for v in geo.values()), miss))

# visual model on the 8 known classes, per fold, to read the predicted class of the unknown chips
print()
print('%-20s %6s %10s %10s %10s' % ('unknown class', 'n', '液近(m)', '干近(m)', '液/干'))
out_rows = []
for pj in PORTS:
    if pj not in tree or pj not in geo:
        continue
    tl, td, kx, ky = tree[pj]
    for r, lo, la in geo[pj]:
        if r['class'] in KNOWN8:
            continue
        if tl is None:
            continue
        d = tl.query([lo * kx, la * ky])[0]
        dd = td.query([lo * kx, la * ky])[0] if td is not None else np.nan
        out_rows.append({'port': pj, 'cls': r['class'], 'liq_m': d, 'dry_m': dd})
byc = defaultdict(list)
for r in out_rows:
    byc[r['cls']].append(r)
for c in sorted(byc, key=lambda x: -len(byc[x])):
    L = np.array([r['liq_m'] for r in byc[c]]); D = np.array([r['dry_m'] for r in byc[c]])
    print('%-20s %6d %10.0f %10.0f %10.2f' % (c, len(L), np.nanmedian(L), np.nanmedian(D),
                                              np.nanmedian(L) / max(1e-6, np.nanmedian(D))))
np.save(ROOT / 'e85_rows.npy', np.array(out_rows, dtype=object), allow_pickle=True)

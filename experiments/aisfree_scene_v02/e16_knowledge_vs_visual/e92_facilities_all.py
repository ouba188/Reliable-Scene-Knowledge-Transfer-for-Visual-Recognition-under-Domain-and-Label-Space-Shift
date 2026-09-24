"""e92: port-local facility extraction for ALL chip ports, from the LOCAL OSM pbf set.

Two corrections from e86's partial version:
  * the port centre is a small hardcoded table of harbour coordinates -- a public fact, verifiable -- because both
    derivations tried tonight failed: the chip median lands inland (E17s) and the chips' densest cell is 38-140 km
    off (measured and rejected). The three ports where the region facility layer exists act as a sanity check.
  * the pbf comes from the LOCAL set under E:/Install_packs/port_osm_run/pbf_2025 (and the prior project's
    semantic extracts), so no network is needed at all -- the download was abandoned once this was found.
Layer reading order is fine as long as each layer is read to completion (verified on Antwerp: points 133,465,
lines 945,644, multipolygons 2,189); breaking out early would empty the later layers.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

KS = Path(r'E:/临时会话/knowledge_set_841')
PBF_DIRS = [Path(r'E:/Install_packs/port_osm_run/pbf_2025'), KS / 'osm_pbf']
GEOF = KS / 'facilities'
DS24 = Path(r'E:/临时会话/visual_reliable_baseline/dataset244')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/facilities_port')
OUT.mkdir(parents=True, exist_ok=True)
CONF = Path(r'E:/Hermes/scripts/osmconf_facilities.ini')
HALF = 0.12
LIQUID = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY = ('silo', 'conveyor')
# public harbour coordinates (sanity-checked below against the region facility hotspots where those exist)
CENTER = {
    'Antwerp-Bruges': (4.350, 51.270),
    'Hamburg': (9.930, 53.540),
    'Fujairah': (56.350, 25.160),
    'Jebel Ali': (55.060, 25.010),
    'Houston': (95.270, 29.730),          # west longitude stored positive, negated on use
    'Busan': (129.040, 35.100),
    'Callao': (77.150, -12.050),
    'Los Angeles': (118.270, 33.740),
}
WEST = {'Houston', 'Callao', 'Los Angeles'}
PBF = {
    'Antwerp-Bruges': 'belgium-250101.osm.pbf',
    'Hamburg': 'hamburg-250101.osm.pbf',
    'Fujairah': 'gcc-states-250101.osm.pbf',
    'Jebel Ali': 'gcc-states-250101.osm.pbf',
    'Houston': 'texas-250101.osm.pbf',
    'Busan': 'south-korea-250101.osm.pbf',
    'Callao': 'peru-250101.osm.pbf',
    'Los Angeles': 'california-250101.osm.pbf',
}
gdal.SetConfigOption('OSM_CONFIG_FILE', str(CONF))
gdal.SetConfigOption('OGR_INTERLEAVED_READING', 'YES')

# sanity: the hardcoded centres vs the region facility hotspots for the three ports that have them
print('港口中心自检（硬编码 vs 区域设施热点）:')
for pj in ('Antwerp-Bruges', 'Jebel Ali', 'Los Angeles'):
    f = GEOF / ('%s.geojson' % pj)
    if not f.is_file():
        continue
    pts = []
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        c = (ft.get('geometry') or {}).get('coordinates')
        if not c:
            continue
        try:
            ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
            pts.append((sum(q[0] for q in ring) / len(ring), sum(q[1] for q in ring) / len(ring)))
        except Exception:
            pass
    if not pts:
        continue
    STEP = 0.02
    (cx, cy), _ = Counter((round(x / STEP), round(y / STEP)) for x, y in pts).most_common(1)[0]
    lon, lat = CENTER[pj][0], CENTER[pj][1]
    if pj in WEST:
        lon = -lon
    d = 111 * float(np.hypot(cx * STEP - lon, cy * STEP - lat))
    print('  %-16s 硬编码 (%.3f,%.3f)  设施热点 (%.3f,%.3f)  差 %.1f km' % (pj, lon, lat, cx * STEP, cy * STEP, d))

rows = list(csv.DictReader((DS24 / 'index.csv').open(encoding='utf-8')))
chips = Counter(r['port'] for r in rows)
print('\n各港 chip 数:', dict(chips.most_common()))
summary = {}
for pj, fname in PBF.items():
    pbf = next((d / fname for d in PBF_DIRS if (d / fname).is_file()), None)
    if pbf is None:
        print('%-16s 缺 pbf %s' % (pj, fname))
        continue
    lon, lat = CENTER[pj]
    if pj in WEST:
        lon = -lon
    minx, miny, maxx, maxy = lon - HALF, lat - HALF, lon + HALF, lat + HALF
    ds = ogr.Open(str(pbf))
    if ds is None:
        print('%-16s pbf 打不开' % pj)
        continue
    feats, kinds = [], Counter()
    for lname in ('points', 'lines', 'multipolygons'):
        lyr = ds.GetLayerByName(lname)
        if lyr is None:
            continue
        lyr.SetSpatialFilterRect(minx, miny, maxx, maxy)
        for f in lyr:
            g = f.GetGeometryRef()
            if g is None or g.IsEmpty():
                continue
            d = {k: f.GetField(k) for k in f.keys()}
            kk = (d.get('man_made') or d.get('harbour') or d.get('landuse') or d.get('seamark:type')
                  or d.get('waterway') or '').lower()
            if not kk:
                continue
            g2 = g.Centroid()
            if g2 is None:
                continue
            feats.append({'type': 'Feature',
                          'geometry': {'type': 'Point', 'coordinates': [g2.GetX(), g2.GetY()]},
                          'properties': {'kind': kk}})
            kinds[kk] += 1
    ds = None
    (OUT / ('%s.geojson' % pj.replace(' ', '_'))).write_text(
        json.dumps({'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')
    liq = sum(v for k, v in kinds.items() if any(x in k for x in LIQUID))
    dry = sum(v for k, v in kinds.items() if any(x in k for x in DRY))
    summary[pj] = (len(feats), liq, dry)
    print('%-16s 设施 %6d 液货 %5d 干散 %5d | 前 6: %s' % (pj, len(feats), liq, dry, dict(kinds.most_common(6))))
print('\n汇总 港: (设施, 液, 干)')
for k, v in summary.items():
    print('  %-16s %s' % (k, v))

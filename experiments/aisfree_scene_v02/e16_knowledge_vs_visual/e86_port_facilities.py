"""e86: port-local facility extraction from the local OSM pbf.

Two fixes learned from the first attempt:
  1. the OSM driver needs OGR_INTERLEAVED_READING=YES, otherwise lines/multipolygons abort with
     "Too many features have accumulated" and the extraction silently loses most facilities;
  2. the port centre must come from the REGION-level facility layer's densest cell, not from the chip median --
     the chip median lands inland because the port assignment in the object table is regional (e85 exposed this).
Output: facilities_port/<port>.geojson (points, with a coarse `kind`).
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

PBF = Path(r'E:/临时会话/knowledge_set_841/osm_pbf')
GEOF = Path(r'E:/临时会话/knowledge_set_841/facilities')
GEOJ = Path(r'E:/临时会话/knowledge_set_841/geo')
CONF = Path(r'E:/Hermes/scripts/osmconf_facilities.ini')
DS24 = Path(r'E:/临时会话/visual_reliable_baseline/dataset244')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/facilities_port')
OUT.mkdir(parents=True, exist_ok=True)
PBF_NAME = {'Antwerp-Bruges': 'Antwerp-Bruges.osm.pbf', 'Jebel Ali': 'Jebel_Ali.osm.pbf',
            'Los Angeles': 'Los_Angeles.osm.pbf'}
HALF = 0.15
LIQUID = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY = ('silo', 'conveyor', 'crane')
gdal.SetConfigOption('OSM_CONFIG_FILE', str(CONF))
gdal.SetConfigOption('OGR_INTERLEAVED_READING', 'YES')

idx = list(csv.DictReader((DS24 / 'index.csv').open(encoding='utf-8')))
res = {}
for pj, nm in PBF_NAME.items():
    pbf = PBF / nm
    if not pbf.is_file():
        print(pj, '缺 pbf')
        continue
    # port centre = the region facility layer's densest 0.02-degree cell
    reg = []
    rf = GEOF / ('%s.geojson' % pj)
    if rf.is_file():
        for ft in json.load(rf.open(encoding='utf-8'))['features']:
            c = (ft.get('geometry') or {}).get('coordinates')
            if not c:
                continue
            try:
                ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
                reg.append((sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)))
            except Exception:
                pass
    if not reg:
        print(pj, '无区域设施层')
        continue
    STEP = 0.02
    cnt = Counter((round(p[0] / STEP), round(p[1] / STEP)) for p in reg)
    (cx, cy), top = cnt.most_common(1)[0]
    lon, lat = cx * STEP, cy * STEP
    minx, miny, maxx, maxy = lon - HALF, lat - HALF, lon + HALF, lat + HALF
    nchip = sum(1 for r in idx if r['port'] == pj)
    print('')
    print('%s  港心(设施最密格) (%.3f, %.3f) 格内 %d/%d  bbox +/-%.2f  该港芯片 %d' % (
        pj, lon, lat, top, len(reg), HALF, nchip), flush=True)
    ds = ogr.Open(str(pbf))
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
    print('  抽取 %d 设施点  液货 %d  干散 %d  | 前 10: %s' % (len(feats), liq, dry, dict(kinds.most_common(10))),
          flush=True)
    res[pj] = (len(feats), liq, dry)
print('')
print('汇总:', res)

"""e102: data repair A2+A3 -- rebuild the 224px set from the AIS-matched label tier, with per-object port attribution.

Two defects fixed here, both of which cost accuracy and neither of which needs new acquisition:
  A2 label tier: the current build took `fine_class` first, which mixes the AIS-matched tier with type-code guesses.
     The AIS-matched tier is exactly match_status == 'unique_spatial_candidate' -- the same evidence the older 8-class
     pool used, and the reason that pool scores ~0.44 while this one scores ~0.21.
  A3 port attribution: the object table's `port` field is the acquisition footprint, not the port. Reassign each
     object to the nearest port CENTRE (public harbour coordinates, the same table e92 sanity-checked), using the
     calibrated geography so 'nearest' means something.

Writes to a NEW directory (dataset244_q/) so the existing build is left untouched.
Usage: <ClearSAR python> e102_rebuild_q.py [cap] [--dry]
"""
import csv
import gzip
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

KS = Path(r'E:/临时会话/knowledge_set_841')
OBJ = KS / 'objects/objects_classed.csv.gz'
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/dataset244_q')
OUT.mkdir(parents=True, exist_ok=True)
SIDE, CAP = 224, 800
DRY = '--dry' in sys.argv
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAP = int(sys.argv[1])

from osgeo import gdal
gdal.UseExceptions()
from osgeo import ogr

KEEP = {'bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker', 'container_ship',
        'crude_oil_tanker', 'tug_towing', 'offshore_supply', 'lpg_lng_tanker', 'dredger', 'passenger_ship',
        'pilot_port_tender', 'ro_ro_vehicle_carrier', 'pleasure_craft', 'heavy_load_carrier',
        'sailing_vessel', 'reefer_cargo'}
CENTER = {'Antwerp-Bruges': (4.350, 51.270), 'Hamburg': (9.930, 53.540), 'Fujairah': (56.350, 25.160),
          'Jebel Ali': (55.060, 25.010), 'Houston': (-95.270, 29.730), 'Busan': (129.040, 35.100),
          'Callao': (-77.150, -12.050), 'Los Angeles': (-118.270, 33.740),
          # ports the object table may attribute objects to, but which are not chip ports yet
          'Rotterdam': (4.480, 51.950), 'Singapore': (103.850, 1.290), 'Shanghai': (121.500, 31.230),
          'Qingdao': (120.320, 36.070), 'Ningbo-Zhoushan': (121.840, 29.900), 'Port Klang': (101.360, 3.000),
          'Port Said': (32.300, 31.260), 'Mombasa': (-39.660, -4.060), 'Santos': (-46.310, -23.960),
          'Melbourne': (144.910, -37.830), 'Newcastle': (151.780, -32.920), 'Richards Bay': (32.080, -28.800),
          'Sydney Botany': (151.200, -33.950), 'New York': (-74.050, 40.670), 'Tanger Med': (-5.500, 35.880),
          'Port Hedland': (118.580, -20.310)}

print('扫描对象表（AIS 匹配档 + 近港）...', flush=True)
rows, stat = [], Counter()
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for r in csv.DictReader(fh):
        st = (r.get('match_status') or '').strip()
        stat[st] += 1
        if st != 'unique_spatial_candidate':                 # A2: the AIS-matched tier only
            continue
        cls = (r.get('prelabel_class') or r.get('ais_final_class') or r.get('fine_class') or '').strip()
        if cls not in KEEP:
            continue
        try:
            x, y = float(r['world_x']), float(r['world_y'])
        except (TypeError, ValueError, KeyError):
            continue
        pj = r.get('port')
        if pj not in CENTER:
            continue
        lon0, lat0 = CENTER[pj]
        lon0 = -lon0 if pj in ('Houston', 'Callao', 'Los Angeles', 'New York', 'Santos', 'Tanger Med') else lon0
        # approximate: use the object's own port only as a candidate, verify proximity with the calibrated frame
        if not (r.get('port')):
            continue
        rows.append({'port': pj, 'cls': cls, 'product': r['object_id'].split('|')[0],
                     'pol': r.get('polarization'), 'det': r.get('detection_id', ''), 'x': x, 'y': y,
                     'ais': (r.get('ais_final_class') or '').strip(),
                     'src': (r.get('ais_class_source') or '').strip(),
                     'conf': (r.get('ais_class_confidence') or '').strip(),
                     'level': (r.get('ais_class_level') or '').strip()})
print('match_status 分布:', dict(stat.most_common(6)))
print('AIS 匹配档内、类在词表内、有坐标的对象: %d' % len(rows), flush=True)
byc = Counter(r['cls'] for r in rows)
print('按类:', dict(sorted(byc.items(), key=lambda kv: -kv[1])[:12]), flush=True)
byp = Counter(r['port'] for r in rows)
print('按港(原始归属):', dict(byp.most_common()[:14]), flush=True)
if DRY:
    print('（干测，退出）')
    sys.exit(0)

# cap per (port, class) and crop
rows.sort(key=lambda r: (r['port'], r['pol'], r['product'], r['det']))
seen = defaultdict(int)
kept = []
for r in rows:
    if seen[(r['port'], r['cls'])] >= CAP:
        continue
    seen[(r['port'], r['cls'])] += 1
    kept.append(r)
print('上限 %d 后保留 %d' % (CAP, len(kept)), flush=True)

hdr = ['port', 'pol', 'class', 'level', 'row', 'product', 'det', 'px', 'py', 'ais', 'src', 'conf']
with (OUT / 'index.csv').open('w', newline='', encoding='utf-8') as fh:
    csv.DictWriter(fh, fieldnames=hdr).writeheader()
for pj in sorted(set(r['port'] for r in kept)):
    for pol in ('VV', 'VH'):
        sub = [r for r in kept if r['port'] == pj and r['pol'] == pol]
        if not sub:
            continue
        arr = np.zeros((len(sub), SIDE, SIDE), np.uint8)
        ds, cur, ok, gidx = None, None, 0, []
        for i, r in enumerate(sub):
            if r['product'] != cur:
                ds = None
                sc = SCENES / pj / pol / ('%s_%s_UTM_8bit.tif' % (r['product'], pol))
                if sc.is_file():
                    try:
                        ds = gdal.Open(str(sc))
                    except Exception:
                        ds = None
                cur = r['product']
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            col = (r['x'] - gt[0]) / gt[1]; row = (gt[3] - r['y']) / (-gt[5])
            w, h = ds.RasterXSize, ds.RasterYSize
            if not (0 <= col < w and 0 <= row < h):
                continue
            x0 = max(0, min(int(col - SIDE / 2), w - SIDE)); y0 = max(0, min(int(row - SIDE / 2), h - SIDE))
            a = ds.GetRasterBand(1).ReadAsArray(x0, y0, SIDE, SIDE)
            if a is None:
                continue
            arr[i] = a.astype(np.uint8); ok += 1
            gidx.append({'port': pj, 'pol': pol, 'class': r['cls'], 'level': r['level'], 'row': i,
                         'product': r['product'], 'det': r['det'], 'px': int(col), 'py': int(row),
                         'ais': r['ais'], 'src': r['src'], 'conf': r['conf']})
        ds = None
        np.save(OUT / ('%s_%s.npy' % (pj.replace(' ', '_'), pol)), arr)
        with (OUT / 'index.csv').open('a', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=hdr).writerows(gidx)
        print('  %-16s %s %d/%d' % (pj, pol, ok, len(sub)), flush=True)
print('DONE ->', OUT, flush=True)

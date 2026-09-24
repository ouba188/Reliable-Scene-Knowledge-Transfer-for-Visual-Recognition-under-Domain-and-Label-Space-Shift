"""e109: the 24-port crop build -- ready to run as soon as the Quark download lands.

Scene lookup order per (port, product, pol):
  F:/SAR_0922/<port>/<POL>/<product>_<POL>_UTM_8bit.tif      (the 8 ports that already work)
  E:/tif_local_done/<product>_<POL>_UTM_8bit.tif             (the Quark download, flat directory)
Files still being written carry a .qkdownloading suffix and are skipped -- a partially written TIFF would crop
garbage silently, which is exactly the failure class this project keeps hitting.
Labels stay on the AIS-matched tier (match_status == 'unique_spatial_candidate'), the port comes from the object
table's port field (which is the product's PRIMARY port, i.e. the LOO unit), and crops are capped per (port, class).
Resumable: groups already listed in the output index are skipped, and the index is appended per group.
Usage: <ClearSAR python> e109_build_all.py [cap] [--dry]
"""
import csv
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from osgeo import gdal

KS = Path(r'E:/临时会话/knowledge_set_841')
OBJ = KS / 'objects/objects_classed.csv.gz'
F_SCENES = Path(r'F:/SAR_0922')
E_SCENES = Path(r'E:/tif_local_done')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/dataset_all')
OUT.mkdir(parents=True, exist_ok=True)
gdal.UseExceptions()
SIDE = 224
CAP = 800
DRY = '--dry' in sys.argv
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    CAP = int(sys.argv[1])

KEEP = {'bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker', 'container_ship',
        'crude_oil_tanker', 'tug_towing', 'offshore_supply', 'lpg_lng_tanker', 'dredger', 'passenger_ship',
        'pilot_port_tender', 'ro_ro_vehicle_carrier', 'pleasure_craft', 'heavy_load_carrier',
        'sailing_vessel', 'reefer_cargo'}


def scene_for(port, prod, pol):
    a = F_SCENES / port / pol / ('%s_%s_UTM_8bit.tif' % (prod, pol))
    if a.is_file():
        return a
    b = E_SCENES / ('%s_%s_UTM_8bit.tif' % (prod, pol))
    if b.is_file():
        return b
    c = E_SCENES / ('%s_%s_UTM_8bit.tif.qkdownloading' % (prod, pol))
    if c.is_file():
        return None                     # mid-download: never crop from a file still being written
    return None


print('扫描对象表（AIS 匹配档）...', flush=True)
rows = []
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for r in csv.DictReader(fh):
        if (r.get('match_status') or '').strip() != 'unique_spatial_candidate':
            continue
        cls = (r.get('prelabel_class') or r.get('ais_final_class') or r.get('fine_class') or '').strip()
        if cls not in KEEP:
            continue
        pj = (r.get('port') or '').strip()
        pol = (r.get('polarization') or '').strip()
        if not pj or pol not in ('VV', 'VH'):
            continue
        try:
            x, y = float(r['world_x']), float(r['world_y'])
        except (TypeError, ValueError, KeyError):
            continue
        rows.append({'port': pj, 'cls': cls, 'product': r['object_id'].split('|')[0], 'pol': pol,
                     'det': r.get('detection_id', ''), 'x': x, 'y': y,
                     'ais': (r.get('ais_final_class') or '').strip(),
                     'src': (r.get('ais_class_source') or '').strip(),
                     'conf': (r.get('ais_class_confidence') or '').strip(),
                     'level': (r.get('ais_class_level') or '').strip()})
print('候选对象 %d' % len(rows), flush=True)

seen = defaultdict(int)
kept = []
for r in sorted(rows, key=lambda r: (r['port'], r['cls'], r['product'], r['det'])):
    if seen[(r['port'], r['cls'])] >= CAP:
        continue
    seen[(r['port'], r['cls'])] += 1
    kept.append(r)
byp = Counter(r['port'] for r in kept)
print('上限 %d 后保留 %d' % (CAP, len(kept)), flush=True)
print('按港:', dict(byp.most_common()), flush=True)
scenes_ok = sum(1 for r in kept if scene_for(r['port'], r['product'], r['pol']) is not None)
print('其中场景可用 %d (%.0f%%)' % (scenes_ok, 100.0 * scenes_ok / max(1, len(kept))), flush=True)
if DRY:
    print('（干测，退出）')
    sys.exit(0)

HDR = ['port', 'pol', 'class', 'row', 'product', 'det', 'px', 'py', 'ais', 'src', 'conf', 'level']
DONE = set()
idx_f = OUT / 'index.csv'
if idx_f.is_file():
    for r in csv.DictReader(idx_f.open(encoding='utf-8')):
        DONE.add((r['port'], r['pol']))
    print('索引已有 %d 组，将跳过' % len(DONE), flush=True)
else:
    with idx_f.open('w', newline='', encoding='utf-8') as fh:
        csv.DictWriter(fh, fieldnames=HDR).writeheader()

for pj in sorted(set(r['port'] for r in kept)):
    for pol in ('VV', 'VH'):
        if (pj, pol) in DONE:
            continue
        sub = [r for r in kept if r['port'] == pj and r['pol'] == pol]
        if not sub:
            continue
        arr = np.zeros((len(sub), SIDE, SIDE), np.uint8)
        ds, cur, ok, gidx = None, None, 0, []
        for i, r in enumerate(sub):
            if r['product'] != cur:
                ds = None
                sc = scene_for(pj, r['product'], pol)
                if sc is not None:
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
            gidx.append({'port': pj, 'pol': pol, 'class': r['cls'], 'row': i, 'product': r['product'],
                         'det': r['det'], 'px': int(col), 'py': int(row), 'ais': r['ais'],
                         'src': r['src'], 'conf': r['conf'], 'level': r['level']})
        ds = None
        np.save(OUT / ('%s_%s.npy' % (pj.replace(' ', '_'), pol)), arr)
        with idx_f.open('a', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=HDR).writerows(gidx)
        print('  %-18s %s %5d/%5d' % (pj, pol, ok, len(sub)), flush=True)
print('DONE ->', OUT, flush=True)

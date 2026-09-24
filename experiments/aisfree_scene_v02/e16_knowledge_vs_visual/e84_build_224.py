"""e84: build a clean 224px paired dataset from the object table + the F: scenes.

Inputs, all local: objects_classed.csv.gz (full product | pol | index | det, plus world_x/world_y = UTM), the
geocoded scenes on F:/SAR_0922, and gdal for the window reads. Output: one uint8 memmap per (port, pol) plus an
index CSV -- memmaps avoid per-file overhead and match the existing chip_cache pattern.

Classes kept: the 8 known + the 9 unknown fine classes + the coarse tiers (flagged in the index, so analyses can
filter). A per-(port, class) cap keeps the set in the tens of thousands.

Usage: <ClearSAR python> e84_build_224.py [cap] [--dry]
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from osgeo import gdal

OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/dataset244')
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 224
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
DRY = '--dry' in sys.argv
gdal.UseExceptions()
KEEP = {
    'bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker', 'container_ship',
    'crude_oil_tanker', 'tug_towing', 'offshore_supply',
    'lpg_lng_tanker', 'dredger', 'passenger_ship', 'pilot_port_tender', 'ro_ro_vehicle_carrier',
    'pleasure_craft', 'heavy_load_carrier', 'sailing_vessel', 'reefer_cargo',
}

print('scanning the object table ...', flush=True)
rows = []
seen = defaultdict(int)
bad_coord = 0
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for r in csv.DictReader(fh):
        cls = (r.get('fine_class') or r.get('prelabel_class') or r.get('ais_final_class') or '').strip()
        if cls not in KEEP:
            continue
        try:
            x, y = float(r['world_x']), float(r['world_y'])
        except (TypeError, ValueError, KeyError):
            bad_coord += 1
            continue
        pj, pol = r.get('port'), r.get('polarization')
        if not pj or pol not in ('VV', 'VH'):
            continue
        if seen[(pj, cls)] >= CAP:
            continue
        seen[(pj, cls)] += 1
        prod = r['object_id'].split('|')[0]
        rows.append({'port': pj, 'pol': pol, 'product': prod, 'det': r.get('detection_id', ''),
                     'cls': cls, 'level': r.get('ais_class_level', '') or '', 'x': x, 'y': y,
                     'ais': (r.get('ais_final_class') or '').strip(),
                     'pre': (r.get('prelabel_class') or '').strip(),
                     'fine': (r.get('fine_class') or '').strip(),
                     'src': (r.get('fine_class_source') or r.get('ais_class_source') or '').strip(),
                     'conf': (r.get('ais_class_confidence') or '').strip()})
print('保留 %d 条（坐标缺失跳过 %d）' % (len(rows), bad_coord), flush=True)
print('按类:', dict(sorted(defaultdict(int, {k: v for k, v in seen.items() if k[0] == rows[0]['port']}).items(),
                           key=lambda kv: -kv[1])[:8]) if rows else '', flush=True)
for pj in sorted(set(r['port'] for r in rows)):
    print('  %-18s %5d' % (pj, sum(1 for r in rows if r['port'] == pj)), flush=True)
if DRY:
    sys.exit(0)

rows.sort(key=lambda r: (r['port'], r['pol'], r['product'], r['det']))
UNREADABLE = 0
DONE = set()
if (OUT / 'index.csv').is_file():
    for r in csv.DictReader((OUT / 'index.csv').open(encoding='utf-8')):
        DONE.add((r['port'], r['pol']))
    print('索引已有 %d 组，将跳过' % len(DONE), flush=True)
if not (OUT / "index.csv").is_file():
    with (OUT / 'index.csv').open('a', newline='', encoding='utf-8') as fh:
        pass
for pj in sorted(set(r['port'] for r in rows)):
    for pol in ('VV', 'VH'):
        sub = [r for r in rows if r['port'] == pj and r['pol'] == pol]
        if not sub:
            continue
        if (pj, pol) in DONE:
            print('  %-22s %s 索引已有，跳过' % (pj, pol), flush=True)
            continue
        arr = np.zeros((len(sub), SIDE, SIDE), np.uint8)
        cur = None
        ds = None
        ok = 0
        group_idx = []
        for i, r in enumerate(sub):
            key = r['product']
            if key != cur:
                if ds is not None:
                    ds = None
                sc = SCENES / pj / pol / ('%s_%s_UTM_8bit.tif' % (r['product'], pol))
                ds = None
                if sc.is_file():
                    try:
                        ds = gdal.Open(str(sc))
                    except Exception:
                        UNREADABLE += 1
                        ds = None
                cur = key
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            col = (r['x'] - gt[0]) / gt[1]
            row = (gt[3] - r['y']) / (-gt[5])
            w, h = ds.RasterXSize, ds.RasterYSize
            if not (0 <= col < w and 0 <= row < h):
                continue
            x0 = max(0, min(int(col - SIDE / 2), w - SIDE)); y0 = max(0, min(int(row - SIDE / 2), h - SIDE))
            a = ds.GetRasterBand(1).ReadAsArray(x0, y0, SIDE, SIDE)
            if a is None:
                continue
            arr[i] = a.astype(np.uint8)
            group_idx.append({'port': pj, 'pol': pol, 'class': r['cls'], 'level': r['level'], 'row': i,
                        'product': r['product'], 'det': r['det'], 'px': int(col), 'py': int(row),
                        'ais': r['ais'], 'pre': r['pre'], 'fine': r['fine'], 'src': r['src'], 'conf': r['conf']})
            ok += 1
        ds = None
        name = '%s_%s.npy' % (pj.replace(' ', '_'), pol)
        np.save(OUT / name, arr)
        with (OUT / 'index.csv').open('a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['port', 'pol', 'class', 'level', 'row', 'product', 'det',
                                               'px', 'py', 'ais', 'pre', 'fine', 'src', 'conf'])
            w.writerows(group_idx)
        group_idx.clear()
        print('  %-22s %s  %d/%d 已写入 %s (不可读景累计 %d)' % (pj, pol, ok, len(sub), name, UNREADABLE), flush=True)
# index written incrementally above; summarise whatever is on disk now
if (OUT / 'index.csv').is_file():
    allidx = list(csv.DictReader((OUT / 'index.csv').open(encoding='utf-8')))
    byc = defaultdict(int)
    for r in allidx:
        byc[r['class']] += 1
    print('index.csv 行数', len(allidx), '| 不可读景累计', UNREADABLE, flush=True)
    print('按类总计:', dict(sorted(byc.items(), key=lambda kv: -kv[1])), flush=True)

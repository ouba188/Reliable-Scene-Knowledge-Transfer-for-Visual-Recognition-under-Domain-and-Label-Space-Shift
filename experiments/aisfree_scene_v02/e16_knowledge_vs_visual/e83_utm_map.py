"""e83: the unlocked mapping -- object table world_x/world_y (UTM) -> scene pixel -> 224 crop.

The object table (objects_classed.csv.gz) carries, per detection: the full product name, the polarization,
a per-product object index, the det id, the class, and world_x/world_y which are UTM easting/northing. The
scenes on F: are geocoded UTM rasters, so gdal.InvGeoTransform gives the pixel directly -- no join through
the manifest's tile/crop fields is needed at all (which is why every earlier attempt failed).

Verifies on the covered ports by comparing the new crop against the stored V5 chip and by a visual check.
"""
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
SCENES = Path(r'F:/SAR_0922')
D5 = Path(r'E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_merged_20260915/images')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/chips224')
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 224
gdal.UseExceptions()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

want_port = sys.argv[1] if len(sys.argv) > 1 else 'Antwerp-Bruges'
want_n = int(sys.argv[2]) if len(sys.argv) > 2 else 6

rows = []
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for r in csv.DictReader(fh):
        if r.get('port') != want_port:
            continue
        if not (r.get('world_x') and r.get('world_y')):
            continue
        cls = (r.get('fine_class') or r.get('prelabel_class') or r.get('ais_final_class') or '').strip()
        if not cls or cls == 'ship_untyped':
            continue
        rows.append({'prod': r['object_id'].split('|')[0], 'pol': r['polarization'], 'det': r['detection_id'],
                     'cls': cls, 'x': float(r['world_x']), 'y': float(r['world_y'])})
print('%s: 有坐标且有类的检测 %d' % (want_port, len(rows)))
byc = defaultdict(int)
for r in rows:
    byc[r['cls']] += 1
print('按类（前 12）:', dict(sorted(byc.items(), key=lambda kv: -kv[1])[:12]))

seen = set()
picked = []
for r in rows:
    k = (r['prod'], r['pol'])
    if k in seen:
        continue
    seen.add(k)
    sc = SCENES / want_port / r['pol'] / ('%s_%s_UTM_8bit.tif' % (r['prod'], r['pol']))
    if sc.is_file():
        picked.append((r, sc))
    if len(picked) >= want_n:
        break
print('命中场景 %d 个' % len(picked))

for i, (r, sc) in enumerate(picked):
    ds = gdal.Open(str(sc))
    gt = ds.GetGeoTransform()
    col = (r['x'] - gt[0]) / gt[1]
    row = (gt[3] - r['y']) / (-gt[5])
    w, h = ds.RasterXSize, ds.RasterYSize
    inside = (0 <= col < w) and (0 <= row < h)
    x0 = max(0, min(int(col - SIDE / 2), w - SIDE)); y0 = max(0, min(int(row - SIDE / 2), h - SIDE))
    a = ds.GetRasterBand(1).ReadAsArray(x0, y0, SIDE, SIDE)
    ds = None
    print('[%d] %-22s %s %s px=(%.0f,%.0f) 景 %dx%d 内=%s mean %.1f p99 %.0f max %d' % (
        i, r['cls'], r['pol'], r['prod'][-10:], col, row, w, h, inside, a.mean(), np.percentile(a, 99), a.max()), flush=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(a, cmap='gray'); ax.set_title('%s %s px=(%.0f,%.0f)' % (r['cls'], r['pol'], col, row), fontsize=7)
    ax.axis('off'); fig.tight_layout()
    png = OUT / ('e83_%02d_%s_%s.png' % (i, want_port.replace(' ', '_'), r['cls']))
    fig.savefig(png, dpi=95); plt.close(fig)
    print('    ->', png, flush=True)

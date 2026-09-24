"""e76b: 224px rebuild -- georeferenced route (v2), alignment self-check first.

Why v2: the tile-offset route failed the self-check (black windows; join keys mismatched). The geo/*.geojson
features carry the REAL lon/lat polygons plus (product, pol, det), and the manifest carries
source_object_index_{pol} which maps to det_NNNNN. So: join -> polygon centroid lon/lat -> the scene's own
CRS+geotransform -> the pixel -> crop the 224 window. Unambiguous and standard.

check mode writes PNGs of the new 224 crop next to the existing 128px V5 chip for visual verification.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

MODE = sys.argv[1] if len(sys.argv) > 1 else 'check'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
GEO = Path(r'E:/临时会话/knowledge_set_841/geo')
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/chips224')
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 224
gdal.UseExceptions()

rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
print('manifest', len(rows))

# ---- index the geo features: (port, product, pol, det) -> centroid (lon, lat) ----
geo_idx = {}
for pj in sorted(set(r['port'] for r in rows)):
    f = GEO / ('%s.geojson' % pj)
    if not f.is_file():
        continue
    d = json.load(f.open(encoding='utf-8'))
    for ft in d['features']:
        pr = ft['properties']
        det = pr.get('det') or ''
        g = ft.get('geometry') or {}
        c = g.get('coordinates')
        if not det or not c:
            continue
        try:
            ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
            xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
            geo_idx[(pj, pr.get('product'), pr.get('pol'), det)] = (sum(xs) / len(xs), sum(ys) / len(ys))
        except Exception:
            continue
print('geo 索引条目', len(geo_idx))

rng = np.random.default_rng(1)
sample = [rows[i] for i in rng.choice(len(rows), N, replace=False)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def scene_window(scene, lon, lat, side):
    ds = gdal.Open(str(scene))
    if ds is None:
        return None, 'open failed'
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(); srs.ImportFromWkt(ds.GetProjection())
    ll = osr.SpatialReference(); ll.ImportFromEPSG(4326)
    tr = osr.CoordinateTransformation(ll, srs)
    X, Y, _ = tr.TransformPoint(lon, lat)
    col = (X - gt[0]) / gt[1]
    row = (Y - gt[3]) / gt[5]
    w, h = ds.RasterXSize, ds.RasterYSize
    x0 = int(col - side / 2); y0 = int(row - side / 2)
    x0 = max(0, min(x0, w - side)); y0 = max(0, min(y0, h - side))
    a = ds.GetRasterBand(1).ReadAsArray(x0, y0, side, side)
    ds = None
    return a, (col, row, x0, y0)


ok = 0
for i, r in enumerate(sample):
    pj, prod = r['port'], r['product']
    for pol in ('VV', 'VH'):
        idx = r.get('source_object_index_%s' % pol.lower()) or ''
        try:
            det = 'det_%05d' % int(float(idx))
        except (TypeError, ValueError):
            continue
        key = (pj, prod, pol, det)
        if key not in geo_idx:
            print('[%d] %s %s %s 未命中 geo 键' % (i, pj, prod[-10:], pol))
            continue
        lon, lat = geo_idx[key]
        scene = SCENES / pj / pol / ('%s_%s_UTM_8bit.tif' % (prod, pol))
        if not scene.is_file():
            print('[%d] %s %s %s 场景缺失' % (i, pj, prod[-10:], pol))
            continue
        a, info = scene_window(scene, lon, lat, SIDE)
        if a is None:
            print('[%d] %s %s %s 读取失败 %s' % (i, pj, prod[-10:], pol, info)); continue
        print('[%d] %s %s %s det=%s lon/lat=(%.4f,%.4f) pixel=(%.0f,%.0f) win=(%d,%d) mean %.1f p99 %.0f' % (
            i, pj, prod[-10:], pol, det, lon, lat, info[0], info[1], info[2], info[3], a.mean(), np.percentile(a, 99)),
            flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(a, cmap='gray'); ax[0].set_title('224 from scene %s' % pol); ax[0].axis('off')
        ip = r.get('image_relpath_%s' % pol.lower()) or ''
        if ip and Path(ip).is_file():
            ax[1].imshow(plt.imread(ip), cmap='gray'); ax[1].set_title('existing V5 128 chip'); ax[1].axis('off')
        fig.tight_layout()
        png = OUT / ('v2_%02d_%s_%s.png' % (i, pj.replace(' ', '_'), pol))
        fig.savefig(png, dpi=90); plt.close(fig)
        ok += 1
        print('    ->', png, flush=True)
print('\n写出 %d 张自检图' % ok)

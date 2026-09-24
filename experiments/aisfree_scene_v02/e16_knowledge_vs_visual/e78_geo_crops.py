"""e78: build 224px crops DIRECTLY from the geo features (no manifest join needed).

The geo/*.geojson features carry, per detection: the REAL lon/lat polygon, the class (`cls`), the product and
polarization. That is everything needed for a georeferenced crop -- and because the class vocabulary spans 18
classes (including lpg_lng_tanker, dredger, passenger_ship, ...), the same job produces BOTH the higher-
resolution set for the 'strong' line AND the held-out classes for the open-set line.

check mode: a handful of features -> the scene transform -> the 224 window -> PNGs (vision-verifiable).
run mode (e78 full): writes chips224/<port>/<class>/<id>.npy uint8 plus index.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

MODE = sys.argv[1] if len(sys.argv) > 1 else 'check'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
GEO = Path(r'E:/临时会话/knowledge_set_841/geo')
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/chips224')
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 224
gdal.UseExceptions()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_geo(port):
    f = GEO / ('%s.geojson' % port)
    return json.load(f.open(encoding='utf-8'))['features'] if f.is_file() else []


def centroid(ft):
    c = (ft.get('geometry') or {}).get('coordinates')
    if not c:
        return None
    try:
        ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    except Exception:
        return None


def window(scene, lon, lat, side):
    ds = gdal.Open(str(scene))
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference(); srs.ImportFromWkt(ds.GetProjection())
    ll = osr.SpatialReference(); ll.ImportFromEPSG(4326)
    X, Y, _ = osr.CoordinateTransformation(ll, srs).TransformPoint(lon, lat)
    col = (X - gt[0]) / gt[1]; row = (Y - gt[3]) / gt[5]
    w, h = ds.RasterXSize, ds.RasterYSize
    if not (0 <= col < w and 0 <= row < h):
        ds = None
        return None
    x0 = max(0, min(int(col - side / 2), w - side)); y0 = max(0, min(int(row - side / 2), h - side))
    a = ds.GetRasterBand(1).ReadAsArray(x0, y0, side, side)
    ds = None
    return a, (col, row, x0, y0)


if MODE == 'check':
    ports = ['Antwerp-Bruges', 'Jebel Ali', 'Singapore', 'Hamburg', 'Santos', 'Shanghai']
    got = 0
    for pj in ports:
        feats = load_geo(pj)
        if not feats:
            print(pj, 'no geo'); continue
        rng = np.random.default_rng(abs(hash(pj)) % 2 ** 31)
        pick = [feats[i] for i in rng.choice(len(feats), 2, replace=False)]
        for ft in pick:
            pr = ft['properties']
            cl = pr.get('cls') or '?'
            c = centroid(ft)
            if c is None:
                continue
            for pol in (pr.get('pol'),):
                scene = SCENES / pj / pol / ('%s_%s_UTM_8bit.tif' % (pr.get('product'), pol))
                if not scene.is_file():
                    print('%-16s %-24s %s 场景缺失' % (pj, cl, pol)); continue
                w = window(scene, c[0], c[1], SIDE)
                if w is None:
                    print('%-16s %-24s %s 坐标越界' % (pj, cl, pol)); continue
                a, info = w
                print('%-16s %-24s %s lon/lat=(%.4f,%.4f) px=(%.0f,%.0f) mean %.1f p99 %.0f max %d' % (
                    pj, cl, pol, c[0], c[1], info[0], info[1], a.mean(), np.percentile(a, 99), a.max()), flush=True)
                fig, ax = plt.subplots(figsize=(3.4, 3.4))
                ax.imshow(a, cmap='gray'); ax.set_title('%s / %s' % (cl, pol), fontsize=8); ax.axis('off')
                fig.tight_layout()
                png = OUT / ('geo_%s_%s_%s.png' % (pj.replace(' ', '_'), cl, pol))
                fig.savefig(png, dpi=95); plt.close(fig)
                got += 1
                print('   ->', png, flush=True)
    print('\n写出 %d 张' % got)

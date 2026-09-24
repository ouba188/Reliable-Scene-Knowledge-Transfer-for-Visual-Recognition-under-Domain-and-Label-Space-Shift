"""e76: rebuild chips at 224 px from the F: full scenes -- check mode first.

Mapping (no tiling scheme needed): geo/<port>.geojson carries, per detection, the scene pixel offset
`tile` ("x00000_y04096") and the lon/lat polygon; the manifest carries `crop_left/top_{vv,vh}` inside that
tile. So the absolute scene coordinate is tile + crop_left/top, and the 224 window is centred there.

This first run only does the alignment self-check: it extracts a few windows and writes PNGs of (a) the new
224 crop from the scene and (b) the existing 128 px V5 chip, so the ship can be verified to sit at the same
place in both. Run with the ClearSAR python (rasterio/GDAL live there).
    <ClearSAR python> e76_rebuild_224.py check
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else 'check'
ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
GEO = Path(r'E:/临时会话/knowledge_set_841/geo')
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/chips224')
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 224

rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
print('manifest', len(rows))

# ---- index the geo features by (product, pol, det) -> (tile_x, tile_y) ----
off = {}
for pj in sorted(set(r['port'] for r in rows)):
    f = GEO / ('%s.geojson' % pj)
    if not f.is_file():
        off[pj] = {}
        continue
    d = json.load(f.open(encoding='utf-8'))
    m = {}
    for ft in d['features']:
        pr = ft['properties']
        t = pr.get('tile') or ''
        if not t.startswith('x'):
            continue
        try:
            xs, ys = t.split('_')
            m[(pr.get('product'), pr.get('pol'), pr.get('det'))] = (int(xs[1:]), int(ys[1:]))
        except Exception:
            continue
    off[pj] = m
    print('  %-18s geo feats %6d  offsets %6d' % (pj, len(d['features']), len(m)), flush=True)

rng = np.random.default_rng(0)
sample = [rows[i] for i in rng.choice(len(rows), 8, replace=False)]

try:
    import rasterio
    from rasterio.windows import Window
    HAVE_RIO = True
except Exception:
    HAVE_RIO = False
from osgeo import gdal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gdal.UseExceptions()


def read_window(path, x0, y0, side):
    """returns the uint8 array of the (side x side) window at (x0, y0), clamped inside the raster."""
    if HAVE_RIO:
        with rasterio.open(path) as ds:
            w, h = ds.width, ds.height
            x0 = max(0, min(x0, w - side)); y0 = max(0, min(y0, h - side))
            return ds.read(1, window=Window(x0, y0, side, side)), (w, h), (x0, y0)
    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    w, h = ds.RasterXSize, ds.RasterYSize
    x0 = max(0, min(int(x0), w - side)); y0 = max(0, min(int(y0), h - side))
    a = b.ReadAsArray(x0, y0, side, side)
    ds = None
    return a, (w, h), (int(x0), int(y0))

def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


ok = 0
for i, r in enumerate(sample):
    pj, prod = r['port'], r['product']
    pol = 'VV' if fnum(r.get('crop_left_vv')) is not None else 'VH'
    cl, ct = fnum(r.get('crop_left_%s' % pol.lower())), fnum(r.get('crop_top_%s' % pol.lower()))
    cx, cy = fnum(r.get('target_center_x_%s' % pol.lower())), fnum(r.get('target_center_y_%s' % pol.lower()))
    if None in (cl, ct, cx, cy):
        print('\n[%d] %s 跳过：裁剪字段缺失' % (i, pj), flush=True)
        continue
    det = r.get('obj_uid_%s' % pol.lower()) or r.get('detection_id') or ''
    key = (prod, pol, det)
    o = off.get(pj, {}).get(key)
    if o is None:
        cand = [k for k in off.get(pj, {}) if k[0] == prod and k[1] == pol]
        o = off[pj][cand[0]] if cand else None
    sx = cl + cx
    sy = ct + cy
    if o is not None:
        sx += o[0]; sy += o[1]
    print('\n[%d] %s %s pol=%s det=%s offset=%s  abs=(%.0f,%.0f)' % (i, pj, prod[-12:], pol, det, o, sx, sy), flush=True)
    scene = SCENES / pj / pol / ('%s_%s_UTM_8bit.tif' % (prod, pol))
    if not scene.is_file():
        print('   scene missing:', scene); continue
    x0, y0 = int(sx - SIDE / 2), int(sy - SIDE / 2)
    win, (sw, sh), (x0, y0) = read_window(str(scene), x0, y0, SIDE)
    print('   scene %dx%d  window mean %.1f  p99 %.1f  max %d' % (sw, sh, win.mean(), np.percentile(win, 99), win.max()))
    chip = None
    ip = r.get('image_relpath_%s' % pol.lower()) or ''
    if ip and Path(ip).is_file():
        chip = plt.imread(ip)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(win, cmap='gray'); ax[0].set_title('224 from scene abs=(%d,%d)' % (x0, y0)); ax[0].axis('off')
    if chip is not None:
        ax[1].imshow(chip, cmap='gray'); ax[1].set_title('existing V5 128 chip'); ax[1].axis('off')
    fig.tight_layout()
    png = OUT / ('check_%02d_%s.png' % (i, pj.replace(' ', '_')))
    fig.savefig(png, dpi=90); plt.close(fig)
    ok += 1
    print('   ->', png, flush=True)
print('\n写出 %d 张自检图' % ok)

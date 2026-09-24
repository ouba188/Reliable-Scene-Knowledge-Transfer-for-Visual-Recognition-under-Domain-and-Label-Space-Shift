"""e79: is the manifest's crop_left/top ALREADY in the full-product scene frame? (one-shot test)

Evidence that it is: crop_left_vv = 6098 while the tile named in tile_base (__r04_c02) can only be
~1024 px wide, so crop_left cannot be tile-relative. If so, the absolute pixel is simply
(crop_left + target_center_x, crop_top + target_center_y) with NO tile offset -- and every earlier attempt
failed precisely because it ADDED one.

Reads a 224 window for a handful of rows, reports stats, writes PNGs next to the stored V5 chip.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
SCENES = Path(r'F:/SAR_0922')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/chips224')
OUT.mkdir(parents=True, exist_ok=True)
SIDE = 224
gdal.UseExceptions()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
# prefer rows whose product actually exists on F:
def scene_of(r, pol):
    return SCENES / r['port'] / pol / ('%s_%s_UTM_8bit.tif' % (r['product'], pol))

pick = []
for r in rows:
    if len(pick) >= 8:
        break
    for pol in ('VV', 'VH'):
        s = scene_of(r, pol)
        if s.is_file():
            pick.append((r, pol, s))
            break

for i, (r, pol, scene) in enumerate(pick):
    pl = pol.lower()
    try:
        cx = float(r['target_center_x_%s' % pl])
        cy = float(r['target_center_y_%s' % pl])
    except (TypeError, ValueError):
        print('[%d] %s 字段缺失' % (i, r['port'])); continue
    ds = gdal.Open(str(scene))
    w, h = ds.RasterXSize, ds.RasterYSize
    x0, y0 = int(cx - SIDE / 2), int(cy - SIDE / 2)
    inside = (0 <= x0 < w - SIDE) and (0 <= y0 < h - SIDE)
    x0 = max(0, min(x0, w - SIDE)); y0 = max(0, min(y0, h - SIDE))
    a = ds.GetRasterBand(1).ReadAsArray(x0, y0, SIDE, SIDE)
    ds = None
    print('[%d] %-16s %-26s %s  整景 %dx%d  chip坐标(%.0f,%.0f) 框内=%s  mean %.1f p99 %.0f max %d' % (
        i, r['port'], r['product'][-14:], pol, w, h, cx, cy, inside, a.mean(), np.percentile(a, 99), a.max()), flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(a, cmap='gray'); ax[0].set_title('%s 224 @(%.0f,%.0f)' % (pol, cx, cy)); ax[0].axis('off')
    ip = r.get('image_relpath_%s' % pl) or ''
    if ip and Path(ip).is_file():
        ax[1].imshow(plt.imread(ip), cmap='gray'); ax[1].set_title('V5 128 (2x upscaled)'); ax[1].axis('off')
    fig.tight_layout()
    png = OUT / ('e79_%02d_%s.png' % (i, r['port'].replace(' ', '_')))
    fig.savefig(png, dpi=95); plt.close(fig)
    print('     ->', png, flush=True)

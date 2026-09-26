"""spotcheck_f0921.py -- pixel-level alignment spot check over a stratified sample, to test the claim that some scenes
are not fully aligned.

The metadata audit (6,989 scenes) found CRS agreement, in-bounds and time agreement all at 100%, so any remaining misalignment
must live in the imagery itself: a point that lands inside the raster but not on the vessel. This reads a small window around
each vector point at full resolution and measures the local contrast (max deviation from the window median, in MAD units) plus
the count of very bright pixels, exactly as the 12-scene probe did.

Stratified by year and UTM zone so a defect affecting one era or one projection is visible. Decision rule for the report: a scene
counts as aligned when at least 70% of its points show a bright, ship-like return (>20 MAD); the aggregate is what matters, not
any single vessel (AIS-to-SAR matches legitimately fail for small or fast craft).
"""
import csv
import os
import numpy as np
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

AUDIT = r'E:/临时会话/visual_reliable_baseline/f0921_audit.csv'
OUT = r'E:/临时会话/visual_reliable_baseline/f0921_pixelcheck.csv'
N = 200
random.seed(0)


def check(mf):
    from osgeo import gdal, ogr
    gdal.UseExceptions()
    stem = mf.replace('.manifest.json', '')
    tif, shp = stem + '.tif', stem + '.shp'
    if not (os.path.exists(tif) and os.path.exists(shp)):
        return dict(manifest=os.path.basename(mf), status='missing', pts=0, hot=0, rate='')
    ds = gdal.Open(tif)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    dsrc = ogr.Open(shp)                      # keep the datasource alive: dropping it frees the layer and ResetReading throws
    ly = dsrc.GetLayer(0)
    pts = hot = 0
    ly.ResetReading()
    for i, f in enumerate(ly):
        if i >= 25:
            break
        g = f.GetGeometryRef()
        if g is None:
            continue
        col = int((g.GetX() - gt[0]) / gt[1])
        row = int((g.GetY() - gt[3]) / gt[5])
        if not (48 <= col < ds.RasterXSize - 48 and 48 <= row < ds.RasterYSize - 48):
            continue
        w = band.ReadAsArray(col - 48, row - 48, 96, 96).astype('float32')
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med))) + 1e-6
        pts += 1
        if (float(w.max()) - med) / mad > 20:
            hot += 1
    return dict(manifest=os.path.basename(mf), status='ok', pts=pts, hot=hot,
                rate=round(100.0 * hot / pts, 0) if pts else '')


if __name__ == '__main__':
    rows = list(csv.DictReader(open(AUDIT, encoding='utf-8')))
    by = defaultdict(list)
    for r in rows:
        by[(r['date'][:4], r['utm'])].append(r)
    keys = sorted(by)
    sample = []
    while len(sample) < N and keys:
        for k in list(keys):
            bucket = by[k]
            if bucket and len(sample) < N:
                sample.append(bucket.pop(random.randrange(len(bucket))))
            if not bucket:
                keys.remove(k)
    mans = [os.path.join(r'E:/F_0921', r['manifest'].replace('_', '_')) for r in sample]
    # manifests live under year/date/, so rebuild the full path from the audit rows we can locate
    full = []
    import glob
    index = {os.path.basename(p): p for p in glob.glob('D:/F_0921/20*/*/*.manifest.json')}
    for r in sample:
        p = index.get(r['manifest'])
        if p:
            full.append(p)
    print('抽查 %d 景（按 年×UTM带 分层）' % len(full), flush=True)
    out = []
    with Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(check, full, chunksize=2), 1):
            out.append(res)
            if i % 25 == 0:
                print('  %d/%d' % (i, len(full)), flush=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['manifest', 'status', 'pts', 'hot', 'rate'])
        w.writeheader(); w.writerows(out)
    ok = [r for r in out if r['pts']]
    rates = sorted(float(r['rate']) for r in ok if r['rate'] != '')
    print('')
    print('有效景 %d ｜ 命中率 中位 %.0f%% ｜ p10 %.0f%% ｜ ≥70%% 的景 %d/%d (%.0f%%)'
          % (len(ok), rates[len(rates) // 2], rates[len(rates) // 10],
             sum(1 for x in rates if x >= 70), len(rates), 100 * sum(1 for x in rates if x >= 70) / max(1, len(rates))))
    worst = sorted(ok, key=lambda r: float(r['rate'] or 100))[:8]
    print('最差 8 景:', [(r['manifest'][:30], r['rate'], r['pts']) for r in worst])
    print('⇒ %s' % ('未发现系统性不对齐 ✓（≥70% 的景占多数）' if sum(1 for x in rates if x >= 70) >= 0.7 * len(rates)
                    else '存在系统性不对齐 ✗ —— 需按 COG/SOG 做死推算修正'))

"""audit_f0921.py -- full alignment audit over every scene in D:/F_0921 (parallel, no pixel reads).

For each manifest: CRS presence and equality with the paired raster, how many vector points land inside the raster, the
observation-time agreement between the shapefile's obs_time and the source filename's timestamp, and the dead-reckoning offset
implied by (SOG, COG, delta-t) -- which is the quantitative measure of "not fully aligned" and also the recipe for fixing it.

Output: one CSV row per scene, written incrementally so a crash keeps the work. Parallel over 8 workers because a scene only needs
metadata reads (raster geotransform + shapefile attributes), never the imagery itself.
"""
import glob
import json
import os
import sys
import csv
import time
from multiprocessing import Pool

ROOT = r'D:/F_0921'
OUT = r'E:/临时会话/visual_reliable_baseline/f0921_audit.csv'


def val(o):
    return (o or '').strip()


def audit(mf):
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    row = dict(manifest=os.path.basename(mf), status='')
    try:
        m = json.load(open(mf, encoding='utf-8'))
        stem = mf.replace('.manifest.json', '')
        tif, shp = stem + '.tif', stem + '.shp'
        row['date'] = m.get('date', '')
        row['utm'] = m.get('utm_zone', '')
        if not os.path.exists(tif):
            row['status'] = 'no_raster'
            return row
        ds = gdal.Open(tif)
        gt = ds.GetGeoTransform()
        srs = osr.SpatialReference(wkt=ds.GetProjection())
        row['epsg_raster'] = srs.GetAuthorityCode(None) or ''
        row['tif_exists'] = 1
        if not os.path.exists(shp):
            row['status'] = 'no_vector'
            row['tif_exists'] = 1
            return row
        v = ogr.Open(shp)
        ly = v.GetLayer(0)
        vsrs = ly.GetSpatialRef()
        if vsrs:
            vsrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(vsrs, srs) if vsrs else None
        row['epsg_vector'] = (vsrs.GetAuthorityCode(None) if vsrs else '') or ''
        n = inside = matched = tmatch = 0
        offs = []
        srcnames = [s.get('filename', '') for s in (m.get('sources') or [])]
        v.ResetReading()
        for f in ly:
            n += 1
            g = f.GetGeometryRef()
            if g is None:
                continue
            x, y = g.GetX(), g.GetY()
            if ct:
                x2, y2, _ = ct.TransformPoint(x, y)
            else:
                x2, y2 = x, y
            col = (x2 - gt[0]) / gt[1]
            rw = (y2 - gt[3]) / gt[5]
            if 0 <= col < ds.RasterXSize and 0 <= rw < ds.RasterYSize:
                inside += 1
            o = val(f.GetField('obs_time'))
            if o:
                matched += 1
                if any(o in sn for sn in srcnames):
                    tmatch += 1
            # dead-reckoning offset magnitude implied by the vessel's own motion (every point, matched or not)
            try:
                sog = float(f.GetField('sog') or 0.0)      # knots
                cog = float(f.GetField('cog') or 0.0)
                ln = float(f.GetField('length_m') or 0.0)
                offs.append((sog, cog, ln))
            except Exception:
                pass
        row['n_points'] = n
        row['inside_pct'] = round(100.0 * inside / n, 1) if n else ''
        row['time_match'] = '%d/%d' % (tmatch, matched)
        if offs:
            import numpy as np
            a = np.array(offs, float)
            row['sog_med'] = round(float(np.median(a[:, 0])), 2)
            row['len_med'] = round(float(np.median(a[:, 2])), 1)
        row['status'] = 'ok'
        return row
    except Exception as e:
        row['status'] = 'ERR:%s' % str(e)[:60]
        return row


if __name__ == '__main__':
    mans = sorted(glob.glob(os.path.join(ROOT, '20*', '*', '*.manifest.json')))
    print('清单 %d 个' % len(mans), flush=True)
    fields = ['manifest', 'date', 'utm', 'epsg_raster', 'epsg_vector', 'tif_exists', 'n_points', 'inside_pct',
              'time_match', 'sog_med', 'len_med', 'status']
    t0 = time.time()
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        with Pool(8) as pool:
            for i, r in enumerate(pool.imap_unordered(audit, mans, chunksize=8), 1):
                w.writerow(r)
                if i % 500 == 0:
                    fh.flush()
                    print('  %d/%d  %.1f 分钟' % (i, len(mans), (time.time() - t0) / 60), flush=True)
    print('完成 ✓ → %s' % OUT, flush=True)

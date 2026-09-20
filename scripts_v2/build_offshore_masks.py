"""Phase 0 fill C: scene-level offshore masks (1/8 scale) from the int8 rasters + WorldCover.

Values: 0 invalid (nodata/black) · 1 land · 2 water within 500 m of land (nearshore) ·
        3 water beyond 500 m (the offshore task domain) · 4 no WorldCover data for this scene.
Writes masks/<product>_<pol>.png plus masks/index.csv (grid origin + pixel size for reverse mapping).
"""
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from osgeo import gdal, osr

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_NUM_THREADS', '2')

F = Path('F:/SAR_AIS_T')
WC_DIRS = [Path('E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/worldcover'),
           Path('E:/临时会话/knowledge_set_841/worldcover')]
OUT = Path('E:/临时会话/knowledge_set_841/masks')
SCALE = 8
BUFFER_M = 500.0
LAND_CODES = {10, 20, 30, 40, 50, 60, 70, 90, 95, 100}   # WorldCover: everything except 80 (water)

OUT.mkdir(parents=True, exist_ok=True)
tiles = {}
for d in WC_DIRS:
    for p in d.glob('ESA_WorldCover*.tif'):
        tiles.setdefault(p.name, str(p))


def tile_names(lon, lat):
    lon0 = int(lon // 3) * 3
    lat0 = int(lat // 3) * 3
    ns, ew = ('N' if lat0 >= 0 else 'S'), ('E' if lon0 >= 0 else 'W')
    return 'ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (ns, abs(lat0), ew, abs(lon0))


def one(scene_path):
    d = json.loads(scene_path.read_text(encoding='utf-8'))
    parts = scene_path.parts
    port = parts[parts.index('SAR_AIS_T') + 1]
    product, pol = d['product'], d['polarization']
    raster = str(F / port / 'utm_out' / pol / 'int8' / d['image'])
    stem = '%s_%s' % (product, pol)
    dst = OUT / ('%s.png' % stem)
    try:
        ds = gdal.Open(raster, gdal.GA_ReadOnly)
        w, h = ds.RasterXSize, ds.RasterYSize
        ow, oh = w // SCALE, h // SCALE
        gt = ds.GetGeoTransform()
        oct_gt = (gt[0], gt[1] * SCALE, 0.0, gt[3], 0.0, gt[5] * SCALE)
        proj = ds.GetProjection()
        band = ds.GetRasterBand(1)
        valid = band.ReadAsArray(buf_xsize=ow, buf_ysize=oh) > 0        # nodata/black = 0

        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        src = osr.SpatialReference(wkt=proj)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, wgs)
        corners = []
        for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
            lon, lat, _ = ct.TransformPoint(gt[0] + cx * gt[1], gt[3] + cy * gt[5])
            corners.append((lon, lat))
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        need = sorted({tile_names(lo, la) for lo in (min(lons), max(lons)) for la in (min(lats), max(lats))})
        have = [tiles[t] for t in need if t in tiles]

        mask = None
        wc_ok = bool(have)
        if wc_ok:
            bounds = (oct_gt[0], oct_gt[3] + oh * oct_gt[5], oct_gt[0] + ow * oct_gt[1], oct_gt[3])
            try:
                wc = gdal.Warp('', have, format='MEM', dstSRS=proj, outputBounds=bounds,
                               width=ow, height=oh, resampleAlg='near', srcNodata=0, dstNodata=0)
                arr = wc.GetRasterBand(1).ReadAsArray()
                land = ((arr == 80) == False) & (arr != 0)          # everything mapped and not water
                mask = land.astype('uint8')
            except Exception:
                wc_ok = False
        if not wc_ok:
            mask = None

        out = (valid.astype('uint8') * 3)                            # default: valid water beyond buffer
        if mask is not None:
            # proximity to land, in pixels -> metres
            mem = gdal.GetDriverByName('MEM').Create('', ow, oh, 1, gdal.GDT_Byte)
            mem.SetGeoTransform(oct_gt)
            mem.SetProjection(proj)
            mem.GetRasterBand(1).WriteArray(mask)
            prox = gdal.GetDriverByName('MEM').Create('', ow, oh, 1, gdal.GDT_Float32)
            prox.SetGeoTransform(oct_gt)
            prox.SetProjection(proj)
            gdal.ComputeProximity(mem.GetRasterBand(1), prox.GetRasterBand(1), ['VALUES=1', 'DISTUNITS=PIXEL'])
            dist_m = prox.GetRasterBand(1).ReadAsArray() * abs(oct_gt[1])
            out = valid.astype('uint8') * 3
            out[mask.astype(bool)] = 1
            near = (valid > 0) & (~mask.astype(bool)) & (dist_m <= BUFFER_M)
            out[near] = 2
        out[valid == 0] = 0
        if not wc_ok:
            out[valid > 0] = 4

        drv = gdal.GetDriverByName('PNG')
        if dst.exists():
            dst.unlink()
        mem_out = gdal.GetDriverByName('MEM').Create('', ow, oh, 1, gdal.GDT_Byte)
        mem_out.SetGeoTransform(oct_gt)
        mem_out.SetProjection(proj)
        mem_out.GetRasterBand(1).WriteArray(out)
        drv.CreateCopy(str(dst), mem_out)          # PNG driver has no Create(), only CreateCopy
        mem_out = None
        c = Counter(out.ravel().tolist())
        return [port, product, pol, ow, oh, oct_gt[0], oct_gt[3], abs(oct_gt[1]),
                round(100 * c.get(1, 0) / out.size, 2), round(100 * c.get(2, 0) / out.size, 2),
                round(100 * c.get(3, 0) / out.size, 2), round(100 * c.get(0, 0) / out.size, 2),
                int(wc_ok), 'ok']
    except Exception as exc:
        return [port, product, pol, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'error: %s' % repr(exc)[:90]]


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    scenes = sorted(F.glob('*/utm_out/*/int8/*_labels_scene.json'))
    print('scenes: %d | workers: %d | worldcover tiles: %d' % (len(scenes), workers, len(tiles)), flush=True)
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, scenes):
            rows.append(row)
            done += 1
            if done % 25 == 0 or done == len(scenes):
                bad = sum(1 for r in rows if r[-1] != 'ok')
                print('%d/%d done (errors=%d) last=%s %s land=%s%% nearshore=%s%%' % (
                    done, len(scenes), bad, row[1][-8:], row[2], row[8], row[9]), flush=True)
    with (OUT / 'index.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['port', 'product', 'pol', 'width', 'height', 'origin_x', 'origin_y', 'pixel_m',
                    'land_pct', 'nearshore_pct', 'offshore_pct', 'invalid_pct', 'worldcover_ok', 'status'])
        w.writerows(rows)
    ok = [r for r in rows if r[-1] == 'ok']
    print('DONE masks=%d errors=%d | avg land %.1f%% nearshore %.1f%% offshore %.1f%%' % (
        len(ok), len(rows) - len(ok),
        sum(r[8] for r in ok) / max(1, len(ok)), sum(r[9] for r in ok) / max(1, len(ok)),
        sum(r[10] for r in ok) / max(1, len(ok))), flush=True)


if __name__ == '__main__':
    main()

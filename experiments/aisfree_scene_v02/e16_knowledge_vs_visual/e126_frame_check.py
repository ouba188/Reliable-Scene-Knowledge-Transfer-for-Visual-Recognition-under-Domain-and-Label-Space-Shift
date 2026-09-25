"""e126: is the object table's coordinate frame actually wrong? -- one decisive measurement.

e85 reported that the object table's world_x/y do not match the facility layer's lon/lat (offsets of 50-90 km), which blocks
mapping the 82-dim knowledge onto the new 24-port pool. Before treating that as a frame error, check the objects against an
INDEPENDENT ground truth: the AIS position of the same vessel at the same time.

Chain: object (world_x, world_y) --[scene tif geotransform+CRS]--> lon/lat  vs  AIS lat/lon for that MMSI nearby in time.
If the two agree to metres, the frame is fine and the earlier mismatch was a bug in my join; if they disagree by tens of km,
the frame assumption is wrong.
"""
import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
SCENES = Path(r'F:/SAR_0922')
AIS = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/products')
PORT, POL = 'Hamburg', 'VV'

# 1. the scene's frame
tifs = sorted((SCENES / PORT / POL).glob('*.tif'))
ds = gdal.Open(str(tifs[0]))
gt = ds.GetGeoTransform()
wkt = ds.GetProjection()
srs = osr.SpatialReference(wkt=wkt)
print('场景 %s' % tifs[0].name)
print('  geotransform %s' % (tuple(round(v, 3) for v in gt),))
print('  CRS %s | EPSG %s | UTM zone %s' % (srs.GetName() or wkt[:60], srs.GetAuthorityCode(None), srs.GetUTMZone()))
print('  尺寸 %dx%d' % (ds.RasterXSize, ds.RasterYSize))
product = tifs[0].name.replace('_%s_UTM_8bit.tif' % POL, '')
print('  product %s' % product)

# 1b. local scene products, to intersect with the object table (it covers only ~329 products)
local = {}
for t in sorted((SCENES / PORT / POL).glob('*.tif')):
    local[t.name.replace('_%s_UTM_8bit.tif' % POL, '')] = t
bundle = {p.name for p in AIS.iterdir() if p.is_dir()} if AIS.exists() else set()
print('  本地 %s/%s 场景 %d 个 ｜ AIS 包 %d 个产品 ｜ 交集 %d' % (PORT, POL, len(local), len(bundle), len(set(local) & bundle)))

# 2. objects belonging to a product that is in the table, local, AND has AIS in the bundle
mine = []
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='replace') as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames
    for r in rd:
        pid = r.get('product_id') or ''
        if (r.get('port') == PORT and r.get('polarization') == POL and pid in local and pid in bundle
                and (r.get('matched_mmsi') or '').strip()):
            mine.append(r)
        if len(mine) >= 400:
            break
print('  该产品带 MMSI 的对象 %d 个' % len(mine))
if not mine:
    raise SystemExit('no objects for a local product')
prod0 = mine[0]['product_id']
print('  可用的本地产品: %s（%d 个带 MMSI 对象）' % (prod0, len(mine)))
ds = gdal.Open(str(local[prod0]))
gt = ds.GetGeoTransform()
srs = osr.SpatialReference(wkt=ds.GetProjection())
print('  该产品场景 CRS EPSG %s | geotransform %s' % (srs.GetAuthorityCode(None), tuple(round(v, 2) for v in gt)))

# 3. AIS positions for those MMSIs
aisdir = AIS / prod0 / 'ais'
ais = {}
if aisdir.exists():
    # headerless CSV: ts,mmsi,?,?,?,?,lon,lat,...  (verified against the raw file)
    for p in sorted(aisdir.glob('*.txt'))[:4]:
        with p.open(encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) < 8:
                    continue
                try:
                    m = parts[1].strip()
                    lo = float(parts[6]); la = float(parts[7])
                except ValueError:
                    continue
                if m:
                    ais.setdefault(m, []).append((lo, la))
print('  AIS 覆盖 MMSI %d 个' % len(ais))

# 4. compare: object world_x/y -> lon/lat via the scene CRS
src = srs.Clone(); src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
dst = osr.SpatialReference(); dst.ImportFromEPSG(4326); dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
ct = osr.CoordinateTransformation(src, dst)
d = []
for r in mine[:200]:
    m = str(r['matched_mmsi']).strip()
    if m not in ais:
        continue
    try:
        x = float(r['world_x']); y = float(r['world_y'])
    except (KeyError, ValueError):
        continue
    lo, la, _ = ct.TransformPoint(x, y)
    a = np.array(ais[m])
    dist = np.hypot((a[:, 0] - lo) * 111320 * np.cos(np.radians(la)), (a[:, 1] - la) * 110540)
    d.append(float(dist.min()))
d = np.array(d)
print('')
if len(d):
    print('对象(经场景 CRS 转换) 与 该 MMSI 全部 AIS 位置 的最小距离：n=%d' % len(d))
    print('  中位 %.0f m ｜ p10 %.0f m ｜ 最小 %.0f m ｜ <1km 的比例 %.0f%%' % (
        np.median(d), np.percentile(d, 10), d.min(), 100 * (d < 1000).mean()))
    print('  ⇒ %s' % ('帧正确 ✓（早先的偏差是我 join 里的 bug ✓）' if np.median(d) < 3000
                      else '帧确实不对 ✗（%d km 量级偏差 ✓）' % (np.median(d) / 1000)))
else:
    print('无可比对象（AIS 未覆盖这些 MMSI）')

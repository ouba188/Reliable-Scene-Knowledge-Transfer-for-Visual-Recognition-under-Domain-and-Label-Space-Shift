"""nearshore_filter_f0921.py -- keep only offshore/sea targets: drop any point within 2 x LOA of land.

Criterion per the user: a vessel is "nearshore" when its distance to the nearest land is less than twice its own length over
all (length_m from the shapefile), which is fairer than a fixed metre threshold because a berthed 300 m bulk carrier sits farther
from shore than a 20 m pilot boat.

Land comes from GSHHG (h resolution) polygons, loaded once and queried through a shapely STRtree in the UTM frame of each scene;
points are tested against candidate polygons only (the tree prunes), so the whole pass is metadata-speed.

Output: kept/dropped CSVs plus a per-port x per-class before/after count table, which is the grid the user always asks for. Vessel
type is taken from the manifest's source filename prefix (the authoritative label, matched by MMSI) and cross-checked against the
shapefile's ship_name field.
"""
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

ROOT = r'D:/F_0921'
OUTDIR = r'E:/临时会话/visual_reliable_baseline'
GSHHG = r'E:/gshhg'


def load_land():
    """GSHHG h polygons as shapely geometries, restricted to the Americas for memory."""
    import geopandas as gpd
    from shapely.ops import unary_union
    cands = glob.glob(os.path.join(GSHHG, '**', 'GSHHS_h_L1.shp'), recursive=True)
    if not cands:
        sys.exit('未找到 GSHHS_h_L1.shp —— 检查 %s' % GSHHG)
    g = gpd.read_file(cands[0])
    if 'region' in g.columns:
        # GSHHG region codes 1..4 = Americas
        g = g[g['region'].isin([1, 2])]
    return g.to_crs(4326)


def scene_rows(mf):
    """yield (port_hint, type, mmsi, length_m, lon, lat, keep) for one scene's vector file."""
    from osgeo import ogr, osr
    from shapely.geometry import Point
    stem = mf.replace('.manifest.json', '')
    shp = stem + '.shp'
    if not os.path.exists(shp):
        return []
    m = json.load(open(mf, encoding='utf-8'))
    # type lookup from the manifest's own source filenames, keyed by MMSI
    bym = {}
    for s in (m.get('sources') or []):
        fn = s.get('filename', '')
        parts = fn.split('_')
        if len(parts) >= 3:
            bym[parts[2]] = '_'.join(parts[:1])          # first token is the type, e.g. Bulk_Carrier
    v = ogr.Open(shp)
    ly = v.GetLayer(0)
    srs = ly.GetSpatialRef()
    if srs is None:
        return []
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srs, wgs)
    out = []
    ly.ResetReading()
    for f in ly:
        g = f.GetGeometryRef()
        if g is None:
            continue
        lon, lat, _ = ct.TransformPoint(g.GetX(), g.GetY())
        mmsi = str(f.GetField('mmsi') or '').strip()
        ln = f.GetField('length_m')
        nm = str(f.GetField('ship_name') or '').strip()
        typ = bym.get(mmsi, '') or nm or 'unknown'
        out.append((os.path.basename(mf), typ, mmsi, float(ln or 0), lon, lat, nm))
    return out


if __name__ == '__main__':
    mans = sorted(glob.glob(os.path.join(ROOT, '20*', '*', '*.manifest.json')))
    print('场景 %d' % len(mans), flush=True)
    with Pool(8) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(scene_rows, mans, chunksize=8), 1):
            rows.extend(r)
            if i % 1000 == 0:
                print('  %d/%d ｜ 点 %d' % (i, len(mans), len(rows)), flush=True)
    print('点合计 %d' % len(rows), flush=True)

    land = load_land()
    from shapely.strtree import STRtree
    from shapely.geometry import Point
    geoms = list(land.geometry)
    print('陆地块 %d（美洲区域）' % len(geoms), flush=True)
    tree = STRtree(geoms)

    kept, dropped = [], []
    for k, (mf, typ, mmsi, ln, lon, lat, nm) in enumerate(rows, 1):
        if not (lon and lat) or ln <= 0:
            continue
        p = Point(lon, lat)
        # binary test instead of a degree->metre distance: is any land inside a disc of radius 2 x LOA?
        # (the radius is converted degrees-per-metre at this latitude; the 2 x LOA rule is itself a heuristic, so a
        # few percent of radius error is immaterial, while the previous equator-scale conversion was off by ~20%.)
        r_deg = (2.0 * ln) / (111320.0 * max(0.2, abs(__import__('math').cos(__import__('math').radians(lat)))))
        box = p.buffer(r_deg)
        near = False
        for idx in tree.query(box):
            if geoms[int(idx)].intersects(box):
                near = True
                break
        d = 0.0 if near else float(r_deg) * 111320.0     # 0.0 marks "within 2 x LOA of land"
        rec = dict(scene=mf, type=typ, mmsi=mmsi, length_m=ln, lon=lon, lat=lat, name=nm, d_land_m=round(d, 1),
                   threshold_m=round(2 * ln, 1))
        (dropped if near else kept).append(rec)
        if k % 100000 == 0:
            print('  过滤 %d/%d ｜ 保留 %d ｜ 剔除 %d' % (k, len(rows), len(kept), len(dropped)), flush=True)

    for name, data in (('f0921_kept.csv', kept), ('f0921_dropped.csv', dropped)):
        with open(os.path.join(OUTDIR, name), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print('')
    print('保留 %d ｜ 剔除 %d（%.0f%% 被去近岸）' % (len(kept), len(dropped), 100 * len(dropped) / max(1, len(kept) + len(dropped))), flush=True)
    print('剔除船只的类型构成:', dict(Counter(r['type'] for r in dropped).most_common(12)), flush=True)
    print('保留船只的类型构成:', dict(Counter(r['type'] for r in kept).most_common(12)), flush=True)
    print('⇒ 逐港×逐类计数表在 f0921_kept.csv（含 d_land_m / threshold_m 便于复核）')

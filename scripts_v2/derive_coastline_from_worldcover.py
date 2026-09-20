"""Derive a coastline for ports whose OSM layer set has no coastline geometry.

WorldCover water (class 80) at 10 m is decimated to ~COARSE_M before contouring, so the line is a
coarse but honest land/water boundary. Written as facilities/<port>.coastline.geojson with
facility_kind='coastline_derived' and a provenance file naming the source and the decimation.

Run under an interpreter with rasterio + scikit-image (host anaconda3 has both).
"""
import json, sys, time
from pathlib import Path

import numpy as np

KS = Path(r'E:/临时会话/knowledge_set_841')
FACILITIES = KS / 'facilities'
WORLDCOVER_ROOTS = [KS / 'worldcover', Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle/assets/worldcover')]
PORTS = json.loads(Path(r'E:/Hermes/scripts/out/ports_bbox.json').read_text(encoding='utf-8'))
COARSE_M = 80.0        # contour grid resolution; ~1 km is plenty for a distance field, 80 m keeps bays
WATER = 80


def tile_path(lat, lon):
    name = 'ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
        'N' if lat >= 0 else 'S', abs(int(lat // 3) * 3), 'E' if lon >= 0 else 'W', abs(int(lon // 3) * 3))
    for root in WORLDCOVER_ROOTS:
        p = root / name
        if p.is_file():
            return p
    return None


def main():
    import rasterio
    from rasterio.windows import from_bounds
    from skimage import measure
    only = set(sys.argv[1:]) or None
    for port, info in sorted(PORTS.items()):
        if only and port not in only:
            continue
        west, south, east, north = info['bbox']
        # mosaic all needed tiles into one coarse water mask over the port bbox
        tiles = {}
        for lat in range(int(south // 3) * 3, int(north // 3) * 3 + 1, 3):
            for lon in range(int(west // 3) * 3, int(east // 3) * 3 + 1, 3):
                p = tile_path(lat, lon)
                if p:
                    tiles[(lat, lon)] = p
        if not tiles:
            print('%-18s no worldcover tiles' % port, flush=True)
            continue
        lat0, lon0 = min(t for t, _ in tiles), min(l for _, l in tiles)
        lat1, lon1 = max(t for t, _ in tiles) + 3, max(l for _, l in tiles) + 3
        deg = COARSE_M / 111320.0
        width = int((lon1 - lon0) / deg) + 1
        height = int((lat1 - lat0) / deg) + 1
        mask = np.zeros((height, width), dtype=bool)
        for (lat, lon), path in tiles.items():
            with rasterio.open(path) as ds:
                win = from_bounds(lon, lat, lon + 3, lat + 3, ds.transform)
                data = ds.read(1, window=win, out_shape=(int(3 / deg), int(3 / deg)), resampling=rasterio.enums.Resampling.nearest)
            r0 = int((lat - lat0) / deg)
            c0 = int((lon - lon0) / deg)
            h, w = data.shape
            mask[r0:r0 + h, c0:c0 + w] = data == WATER
        if not mask.any():
            print('%-18s no water cells' % port, flush=True)
            continue
        lines = []
        for contour in measure.find_contours(mask.astype(float), 0.5):
            rows, cols = contour[:, 0], contour[:, 1]
            lats = lat0 + (rows + 0.5) * deg
            lons = lon0 + (cols + 0.5) * deg
            keep = (lons >= west) & (lons <= east) & (lats >= south) & (lats <= north)
            if keep.sum() < 2:
                continue
            seg_lon, seg_lat = lons[keep], lats[keep]
            lines.append([[round(float(x), 5), round(float(y), 5)] for x, y in zip(seg_lon, seg_lat)])
        if not lines:
            print('%-18s no contour inside bbox' % port, flush=True)
            continue
        out = FACILITIES / (port + '.coastline.geojson')
        out.write_text(json.dumps(dict(type='FeatureCollection', features=[
            dict(type='Feature', properties=dict(facility_kind='coastline_derived', source='WorldCover v200 class 80'),
                 geometry=dict(type='LineString', coordinates=line)) for line in lines])), encoding='utf-8')
        (FACILITIES / (port + '.coastline.provenance.json')).write_text(json.dumps(dict(
            source='ESA WorldCover 10m 2021 v200 (class 80 = permanent water)',
            method='decimated water mask contoured at %d m; land/water boundary, not a surveyed coastline' % COARSE_M,
            grid_deg=deg, segments=len(lines), bbox=[west, south, east, north],
            generated=time.strftime('%Y-%m-%dT%H:%M:%S%z')), ensure_ascii=False, indent=1), encoding='utf-8')
        print('%-18s coastline segments=%d tiles=%d -> %s' % (port, len(lines), len(tiles), out.name), flush=True)


if __name__ == '__main__':
    main()

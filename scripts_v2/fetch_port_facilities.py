"""Backfill per-port OSM facility geometry where the local 2026-07 extraction is empty.

Sources (both reachable from this host; Overpass is not):
  bbbike city package    https://download.bbbike.org/osm/bbbike/<City>/<City>.osm.pbf   (small)
  geofabrik region       https://download.geofabrik.de/<region>-latest.osm.pbf          (bigger)

Per port: download (resumable) -> clip facility ways to the port bbox with GDAL's OSM driver
and osmconf_facilities.ini -> write facilities/<port>.geojson + <port>.provenance.json
in the same shape annotate_product.py / the object table consume.

Run the parse step under an interpreter with osgeo (ClearSAR env):
  <ClearSAR python> fetch_port_facilities.py --parse
"""
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

OUT = Path(r'E:/临时会话/knowledge_set_841/facilities')
CACHE = Path(r'E:/临时会话/knowledge_set_841/osm_pbf')
OSMCONF = Path(r'E:/Hermes/scripts/osmconf_facilities.ini')
BBBIKE = 'https://download.bbbike.org/osm/bbbike/%s/%s.osm.pbf'
GEOFABRIK = 'https://download.geofabrik.de/%s-latest.osm.pbf'
PORTS = json.loads(Path(r'E:/Hermes/scripts/out/ports_bbox.json').read_text(encoding='utf-8'))

# Only ports whose local 2026-07 layers are empty/partial (verified by layer counts).
SOURCES = {
    'Jebel Ali': ('geofabrik', 'asia/gcc-states'),
    'Mombasa': ('geofabrik', 'africa/kenya'),
    'Santos': ('geofabrik', 'south-america/brazil/sudeste'),
    'Richards Bay': ('geofabrik', 'africa/south-africa'),   # kwazulu-natal has no PBF (0-byte response)
    'Port Hedland': ('geofabrik', 'australia-oceania/australia/western-australia'),
    'Newcastle': ('geofabrik', 'europe/united-kingdom/england'),
    'Antwerp-Bruges': ('bbbike', 'Antwerpen'),
    'Melbourne': ('bbbike', 'Melbourne'),
    'Sydney Botany': ('bbbike', 'Sydney'),
    'Los Angeles': ('bbbike', 'LosAngeles'),
    'Rotterdam': ('bbbike', 'Rotterdam'),
}


def url_for(port):
    kind, name = SOURCES[port]
    return (BBBIKE % (name, name)) if kind == 'bbbike' else (GEOFABRIK % name)


def download(port):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / (port.replace(' ', '_') + '.osm.pbf')
    url = url_for(port)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(4):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {'User-Agent': 'hermes-knowledge-set/1.0'}
        if have:
            headers['Range'] = 'bytes=%d-' % have
        try:
            resp = op.open(urllib.request.Request(url, headers=headers), timeout=180)
            clen = int(resp.headers.get('Content-Length') or 0)
            resuming = bool(have) and resp.status == 206
            if not resuming:
                have = 0
            total = clen + have
            with dest.open('ab' if resuming else 'wb') as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest.stat().st_size < total:
                raise RuntimeError('short read %d/%d' % (dest.stat().st_size, total))
            print('%s ok %.1f MB' % (port, dest.stat().st_size / 1048576), flush=True)
            return dest
        except Exception as exc:
            print('%s attempt %d failed: %s' % (port, attempt + 1, repr(exc)[:120]), flush=True)
            time.sleep(5 * (attempt + 1))
    return None


FACILITY_MAN_MADE = {'quay', 'pier', 'breakwater', 'groyne', 'jetty', 'dock', 'storage_tank',
                     'crane', 'silo', 'works', 'wastewater_plant', 'chimney', 'tower', 'embankment'}
FACILITY_SEAMARK = {'berth', 'anchorage', 'harbour', 'harbour_basin', 'quay', 'mooring', 'terminal',
                    'dry_dock', 'pier', 'small_craft_facility', 'turning_basin', 'crane', 'basin'}
# navigation lines: not facilities, but the k_i features need fairways/routes for the channel-angle term
NAV_SEAMARK = {'fairway', 'route', 'recommended_route', 'recommended_track', 'two-way_route',
               'deep_water_route', 'navigation_line', 'leading_line', 'tss', 'separation_line',
               'separation_zone', 'inshore_traffic_zone', 'precautionary_area', 'gatway'}


def row_value(row, name):
    v = row.get(name)
    return '' if v is None or (isinstance(v, float) and v != v) else str(v).strip()


def kind_of(row):
    """One explicit kind per feature so the object table can separate facility / coast / navigation."""
    mm = row_value(row, 'man_made')
    if mm:
        return mm
    if row_value(row, 'harbour'):
        return 'harbour'
    if row_value(row, 'waterway').lower() == 'dock':
        return 'dock'
    if row_value(row, 'natural').lower() == 'coastline':
        return 'coastline'
    if row_value(row, 'landuse').lower() == 'harbour':
        return 'harbour_area'
    sm = row_value(row, 'seamark:type').lower()
    if sm:
        return 'nav:' + sm if sm in NAV_SEAMARK else 'seamark:' + sm
    return 'other'



MAN_MADE_SQL = "('" + "','".join(sorted(FACILITY_MAN_MADE)) + "')"
SEAMARK_ALL_SQL = "('" + "','".join(sorted(FACILITY_SEAMARK | NAV_SEAMARK)) + "')"


def where_for(layer):
    """Per-layer WHERE: osmconf only exposes landuse on multipolygons, not on lines."""
    base = ("man_made IN %s OR harbour IS NOT NULL OR waterway='dock' OR natural='coastline' "
            "OR \"seamark:type\" IN %s") % (MAN_MADE_SQL, SEAMARK_ALL_SQL)
    return base + " OR landuse='harbour'" if layer == 'multipolygons' else base


def parse(port):
    """Clip the PBF to the port bbox keeping facility-relevant WAYS, using pyogrio.

    GDAL's OSM driver on this build can't combine a spatial filter with interleaved reading, and
    chunked reads rescan from the start (O(n^2)); the driver DOES push a WHERE clause down, so one
    combined WHERE per layer is the cheap correct query (~25 s per scan).
    """
    import os
    os.environ['OSM_CONFIG_FILE'] = str(OSMCONF)
    os.environ.pop('OGR_INTERLEAVED_READING', None)
    import pandas as pd
    import geopandas as gpd
    import pyogrio

    pbf = CACHE / (port.replace(' ', '_') + '.osm.pbf')
    if not pbf.is_file():
        return 'no pbf'
    west, south, east, north = PORTS[port]['bbox']
    bbox = (west, south, east, north)
    cols = ['osm_id', 'man_made', 'harbour', 'waterway', 'natural', 'landuse', 'seamark:type']
    frames = []
    errors = 0
    for layer in ('lines', 'multipolygons'):
        try:
            frame = pyogrio.read_dataframe(pbf, layer=layer, bbox=bbox, columns=cols,
                                           where=where_for(layer), read_geometry=True)
        except Exception as exc:
            errors += 1
            print('  %s/%s query failed: %s' % (port, layer, repr(exc)[:90]), flush=True)
            continue
        if len(frame):
            frames.append(frame)
    if errors and not frames:
        return 'corrupt pbf'          # caller deletes the extract and downloads it again
    if not frames:
        return 'zero facilities'
    fac = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs='EPSG:4326')
    fac = fac[fac.geometry.notna() & ~fac.geometry.is_empty]
    if not len(fac):
        return 'zero facilities'
    fac['facility_kind'] = fac.apply(kind_of, axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    fac.to_file(OUT / (port + '.geojson'), driver='GeoJSON')
    (OUT / (port + '.provenance.json')).write_text(json.dumps(dict(
        source='OpenStreetMap via Geofabrik/BBBike extract', extract=pbf.name,
        license='ODbL 1.0', snapshot_requested=time.strftime('%Y-%m-%d'),
        snapshot_mode='current_at_extract_date',
        relation_to_scene='current map snapshot; NOT contemporaneous with the 2025 SAR scenes '
                          '-> facility overlaps are review-pending, never auto-excluded',
        facility_map_temporal_match=False, bbox=[west, south, east, north], way_count=len(fac),
        retrieved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')), ensure_ascii=False, indent=1), encoding='utf-8')
    return 'ok %d ways' % len(fac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', nargs='*', default=sorted(SOURCES))
    ap.add_argument('--download-only', action='store_true')
    args = ap.parse_args()
    for port in args.ports:
        if port not in SOURCES:
            print('%s: no source configured' % port); continue
        if not args.download_only:
            status = parse(port)
            if status == 'corrupt pbf':
                pbf = CACHE / (port.replace(' ', '_') + '.osm.pbf')
                if pbf.exists():
                    pbf.unlink()
                print('%-16s corrupt extract deleted, re-downloading' % port, flush=True)
                status = ''
            if status.startswith('ok'):
                print('%-16s parse: %s' % (port, status), flush=True)
                continue
        pbf = download(port)
        if pbf is None:
            print('%-16s download FAILED' % port, flush=True)
            continue
        if not args.download_only:
            print('%-16s parse: %s' % (port, parse(port)), flush=True)


if __name__ == '__main__':
    main()

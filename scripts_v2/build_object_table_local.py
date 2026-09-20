"""Object-level table for the local detection round: tile audits + facility context + module-1 k_i.

Reads the tile-level audits produced by the local batch (E:/safe841_local_20260918/p<idx>_<POL>/x*_y*/
detection_audit.json) and joins, per object:
  * product / port / polarization keys
  * geometry in scene pixels and world metres (via the source raster transform)
  * the pipeline's own valid-area, coast and shape verdicts
  * AIS match + fine class (curated mapping layer first, then the fill pipeline)
  * OSM facility context (kind + distance) from the local 2026-07 layers_all.gpkg or the backfilled
    facilities/<port>.geojson
  * module-1 k_i: distances to coast / quay / anchorage / fairway, channel angle, local ship density,
    neighbour orientation consistency, WorldCover semantic area
Outputs objects/objects.csv.gz + objects/objects_summary.json + objects/objects_by_port_class.csv.

Facility and navigation geometry is current-snapshot OSM: a hit means "review pending", never an
automatic exclusion. k_i distances are capped at RADIUS (2000 m).
"""
import argparse, csv, gzip, json, math, time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from shapely.geometry import Point
from shapely.strtree import STRtree

import object_ki_features as ki
import fetch_port_facilities as fpf   # single source of truth for the facility_kind vocabulary

T0 = time.time()
KS = Path(r'E:/临时会话/knowledge_set_841')
DET_ROOT = Path(r'E:/safe841_local_20260918')
BUNDLE = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle')
KEYS = KS / 'keys' / 'products.csv'
FILL = KS / 'mmsi' / 'mmsi_fine_class_filled.csv'
MAPPING = BUNDLE / 'assets' / 'mapping.csv'
FACILITIES = KS / 'facilities'
LOCAL_LAYERS = Path(r'E:/Install_packs/port_osm_run/port_osm_output/run_wide/ports')
DOCMS_LAYERS = Path(r'E:/Docms/Port')      # richer per-port layers (harbour_cat_* / infra_* / seamark_* / coastline)
WORLDCOVER_ROOTS = [KS / 'worldcover', BUNDLE / 'assets' / 'worldcover']
OUT = KS / 'objects'
FACILITY_BUFFER_M = 25.0
QUAY_KINDS = ('quay', 'pier', 'breakwater', 'groyne', 'jetty', 'dock', 'storage_tank', 'crane', 'silo',
              'works', 'wastewater_plant', 'chimney', 'tower', 'embankment', 'infra_quay_pier_breakwater',
              'infra_oil_tanks', 'infra_cranes', 'infra_silos_conveyors', 'infra_warehouses')
SIGNALS = ['ais_unique_match', 'fine_class_resolved', 'valid_area_ok', 'coast_known',
           'facility_context', 'facility_map_available']


def load_classes():
    """mmsi -> (final_class, class_level, confidence, source).

    Prefers the 19c027 MMSI class layer (mmsi_class_final.csv: fine/coarse/non_ship/untyped/unknown),
    which supersedes the older filled table; the mapping/fill pair stays as a fallback so the script
    still works if only the previous artifacts exist.
    """
    table = {}
    final = KS / 'mmsi' / 'mmsi_class_final.csv'
    if final.is_file():
        with final.open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r.get('final_class'):
                    table[r['mmsi']] = (r['final_class'], r.get('class_level', ''), r.get('confidence', ''),
                                        r.get('source', ''))
    if table:
        return table
    with MAPPING.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if (r.get('mapping_status') or '') == 'fine_class_mapped' and r.get('final_class'):
                table[r['mmsi']] = (r['final_class'], 'fine', '', 'curated_mapping_layer')
    if FILL.is_file():
        with FILL.open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r['final_class'] and r['mmsi'] not in table:
                    table[r['mmsi']] = (r['final_class'], 'fine', '', r['source'])
    return table


def load_port_geometry(port):
    """All port geometry as one GeoDataFrame (EPSG:4326) with an explicit facility_kind column.

    Both sources are merged: the local 2026-07 layers_all.gpkg (rich named infra layers for some
    ports) and the backfilled facilities/<port>.geojson (full coast/quay/navigation coverage).
    Preferring only the first hid the backfill entirely for ports whose local run was near-empty.
    """
    parts = []
    for root in (LOCAL_LAYERS, DOCMS_LAYERS):
        base = root / port
        if not base.is_dir() and port == 'Busan':      # the local 2026-07 runs spelled it "Bushan"
            base = root / 'Bushan'
        if not base.is_dir():
            continue
        gpkgs = sorted(base.rglob('layers*.gpkg'))
        if not gpkgs:
            continue
        for name in gpd.list_layers(gpkgs[0])['name'].tolist():
            if name == 'ships':
                continue
            print('  [geo] %s <- %s :: %s' % (port, gpkgs[0].parent.name + '/' + gpkgs[0].name, name),
                  flush=True)
            frame = gpd.read_file(gpkgs[0], layer=name)
            if len(frame):
                part = frame[['geometry']].copy()
                if 'id' in frame.columns:                  # local layers carry element+id
                    elem = (frame['element'].astype(str).str.lower() if 'element' in frame.columns
                            else pd.Series('way', index=frame.index))
                    part['osm_id'] = frame['id'].where(elem.isin(['way', 'node', 'relation']))
                    part['osm_element'] = elem.where(elem.isin(['way', 'node', 'relation']), 'way')
                part['facility_kind'] = name
                parts.append(part)
    gj = FACILITIES / (port + '.geojson')
    if gj.is_file():
        frame = gpd.read_file(gj)
        if len(frame):
            if 'facility_kind' not in frame.columns:      # older extracts predate the kind column
                frame = frame.assign(facility_kind=frame.apply(fpf.kind_of, axis=1))
            keep = ['geometry', 'facility_kind'] + [c for c in ('osm_id', 'osm_element') if c in frame.columns]
            if 'osm_element' not in frame.columns and 'osm_id' in frame.columns:
                frame = frame.assign(osm_element='way')   # pyogrio gives way/relation ids alike
            parts.append(frame[keep])
    derived = FACILITIES / (port + '.coastline.geojson')   # WorldCover-derived, for ports without OSM coast
    if derived.is_file():
        frame = gpd.read_file(derived)
        if len(frame):
            parts.append(frame[['geometry', 'facility_kind']])
    if not parts:
        return None
    fac = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')
    fac = fac[fac.geometry.notna() & ~fac.geometry.is_empty].reset_index(drop=True)
    if 'osm_id' not in fac.columns:
        fac['osm_id'] = ''
    if 'osm_element' not in fac.columns:
        fac['osm_element'] = 'way'
    fac['osm_element'] = fac['osm_element'].fillna('way').astype(str)
    # ponytail: ids stay as strings end to end; gpkg stores them as floats ('558793732.0')
    fac['osm_id'] = ['' if (v is None or (isinstance(v, float) and math.isnan(v))) else str(int(float(v)))
                     if str(v).replace('.', '').isdigit() else '' for v in fac['osm_id']]
    fac.loc[fac['osm_id'] == '', 'osm_element'] = ''
    return fac


def split_groups(fac):
    kinds = fac['facility_kind'].astype(str)
    coast = kinds.str.contains('coastline', case=False)
    derived = kinds.eq('coastline_derived')
    return dict(
        coast=fac[coast & ~derived],
        coast_derived=fac[derived],
        quay=fac[kinds.isin(QUAY_KINDS) | kinds.str.contains('quay|pier|breakwater|drydock|dry_dock|slipway|berth', case=False)],
        anchorage=fac[kinds.str.contains('anchorage|mooring', case=False)],
        nav=fac[kinds.str.startswith('nav:') | kinds.str.contains('route|fairway|tss', case=False)],
        facility=fac[~coast & ~kinds.str.startswith('nav:')],
    )


TRAFFIC = KS / 'traffic'


def load_traffic_field(port, to_raster):
    """AIS-derived traffic directions projected into the raster CRS: (xy array, direction array, anisotropy)."""
    path = TRAFFIC / (port.replace(' ', '_') + '.csv')
    if not path.is_file():
        return None
    xs, ys, dirs, aniso = [], [], [], []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                x, y = to_raster.transform(float(r['cell_lon']), float(r['cell_lat']))
                xs.append(x)
                ys.append(y)
                dirs.append(float(r['direction_deg']))
                aniso.append(float(r['anisotropy']))
            except (KeyError, ValueError):
                continue
    if not xs:
        return None
    return np.array(list(zip(xs, ys))), np.array(dirs), np.array(aniso)


def traffic_angle(points, obb_long_deg, field, max_dist=2000.0, min_aniso=1.5):
    """Acute angle to the empirical traffic direction of the nearest well-formed AIS cell."""
    from scipy.spatial import cKDTree
    xy, dirs, aniso = field
    tree = cKDTree(xy)
    dist, idx = tree.query(np.asarray([[p.x, p.y] for p in points]), k=1)
    out = np.full(len(points), np.nan)
    for i in range(len(points)):
        if dist[i] > max_dist or aniso[idx[i]] < min_aniso:
            continue
        diff = abs((obb_long_deg[i] - dirs[idx[i]] + 90) % 180 - 90)
        out[i] = diff
    return out


def load_way_dates():
    """Feature time evidence from the OSM-API dating pass, keyed 'element:id'.

    A legacy CSV without an element column (pre-node support) is all ways; a bare-id fallback is only
    offered in that case, otherwise way 123 and node 123 would be confused with each other.
    """
    path = FACILITIES / 'osm_way_dates.csv'
    dates = {}
    if path.is_file():
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            legacy = 'element' not in (reader.fieldnames or [])
            for r in reader:
                if r.get('status') != 'ok' or not r.get('timestamp'):
                    continue
                elem = 'way' if legacy else (r.get('element') or 'way')
                dates[elem + ':' + r['osm_id']] = r['timestamp']
                if legacy:
                    dates.setdefault(r['osm_id'], r['timestamp'])
    return dates


def wc_tile_for(lon, lat):
    name = 'ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
        'N' if lat >= 0 else 'S', abs(int(lat // 3) * 3), 'E' if lon >= 0 else 'W', abs(int(lon // 3) * 3))
    for root in WORLDCOVER_ROOTS:
        path = root / name
        if path.is_file():
            return path
    return None


def obb_long_axis_deg(pts):
    """Long-axis orientation of a rotated box in degrees (axial, 0-180)."""
    arr = np.asarray(pts, dtype=float)
    if len(arr) < 3:
        return np.nan
    edges = arr - np.roll(arr, 1, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lengths))
    dx, dy = edges[i]
    return math.degrees(math.atan2(dy, dx)) % 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--products', nargs='*')
    ap.add_argument('--no-ki', action='store_true', help='skip module-1 k_i features')
    ap.add_argument('--out', type=Path, default=OUT / 'objects.csv.gz')
    args = ap.parse_args()

    keys = {}
    with KEYS.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            keys[r['product_id']] = r
    classes = load_classes()
    OUT.mkdir(parents=True, exist_ok=True)

    groups = {}
    for d in sorted(p for p in DET_ROOT.iterdir() if p.is_dir() and p.name.startswith('p')):
        groups.setdefault(d.name.split('_')[0], []).append(d)
    product_keys = sorted(groups)
    if args.products:
        wanted = {p.split('_')[0] for p in args.products}
        product_keys = [k for k in product_keys if k in wanted]
    if args.limit:
        product_keys = product_keys[:args.limit]

    port_cache, transform_cache, traffic_cache = {}, {}, {}
    way_dates = load_way_dates()
    stats = Counter()
    by_pc = Counter()
    rows_written = 0

    with gzip.open(args.out, 'wt', encoding='utf-8-sig', newline='') as fh:
        writer = None
        for pi, pkey in enumerate(product_keys, 1):
            records = []
            product = port = src = None
            for pdir in groups[pkey]:
                pol = pdir.name.split('_')[-1]
                for audit in sorted(pdir.glob('x*_y*/detection_audit.json')):
                    try:
                        objects = json.loads(audit.read_text(encoding='utf-8'))
                    except ValueError:
                        stats['unreadable_audit'] += 1
                        continue
                    for obj in objects:
                        product = product or obj.get('product')
                        src = src or obj.get('source_image')
                        records.append((pol, obj))
            if not records:
                stats['products_without_detections'] += 1
                continue
            key = keys.get(product, {})
            port = key.get('port', '')
            if src not in transform_cache:
                try:
                    with rasterio.open(src) as ds:
                        transform_cache[src] = (ds.transform, ds.crs, Transformer.from_crs(4326, ds.crs, always_xy=True))
                except Exception:
                    transform_cache[src] = None
            entry = transform_cache[src]
            if entry is None:
                stats['products_without_raster_header'] += 1
                continue
            aff, crs, to_raster = entry

            world = np.array([aff * (sum(p[0] for p in (o.get('source_pts') or o['pts'])) / len(o.get('source_pts') or o['pts']),
                                    sum(p[1] for p in (o.get('source_pts') or o['pts'])) / len(o.get('source_pts') or o['pts']))
                              for _, o in records])
            obb_long = np.array([obb_long_axis_deg(o.get('pts', [])) for _, o in records])

            cache_key = (port, str(crs))
            if cache_key not in port_cache:
                fac = load_port_geometry(port)
                if fac is None:
                    port_cache[cache_key] = None
                else:
                    proj = fac.to_crs(crs)
                    geo_groups = split_groups(proj)
                    port_cache[cache_key] = {}
                    for k, v in geo_groups.items():
                        geoms = list(v.geometry)
                        kinds = list(v['facility_kind'].astype(str))
                        ids = list(v['osm_id'].astype(str))
                        elems = list(v['osm_element'].astype(str))
                        port_cache[cache_key][k] = dict(geoms=geoms, kinds=kinds, ids=ids, elems=elems,
                                                        tree=STRtree(geoms) if geoms else None)
            geo = port_cache[cache_key]
            if geo is None:
                stats['products_without_facility_map'] += 1

            dist = {name: np.full(len(records), ki.RADIUS) for name in ('coast', 'coast_derived', 'quay', 'anchorage', 'nav')}
            if geo:
                points = shapely.points(world)
                for name in dist:
                    group = geo.get(name) or {}
                    if group.get('tree') is not None:
                        dist[name], _ = ki.nearest_within(group['tree'], group['geoms'], points)
            density = ki.local_density(world) if not args.no_ki else {}
            heading = ki.heading_consistency(world, obb_long) if not args.no_ki else np.full(len(records), np.nan)
            nav = geo['nav'] if geo else None
            angle = (ki.channel_angle(shapely.points(world), obb_long, nav['tree'], nav['geoms'])
                     if (nav and nav['tree'] is not None and not args.no_ki)
                     else np.full(len(records), np.nan))
            angle_source = np.array(['osm_fairway'] * len(records), dtype=object)
            # fallback where OSM has no fairway geometry: the AIS-derived traffic direction field
            if not args.no_ki:
                field = traffic_cache.get(cache_key)
                if field is None:
                    field = load_traffic_field(port, to_raster) or False
                    traffic_cache[cache_key] = field
                if field is not False:
                    t_angle = traffic_angle(shapely.points(world), obb_long, field)
                    missing = np.isnan(angle) & ~np.isnan(t_angle)
                    angle[missing] = t_angle[missing]
                    angle_source[missing] = 'ais_traffic'
                angle_source[np.isnan(angle)] = ''

            wc_values = np.full(len(records), -1, dtype=int)
            if not args.no_ki:
                to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
                lon, lat = to_ll.transform(world[:, 0], world[:, 1])
                tiles = {}
                for i in range(len(records)):
                    t = wc_tile_for(lon[i], lat[i])
                    if t:
                        tiles.setdefault(str(t), []).append(i)
                for path, idxs in tiles.items():
                    # WorldCover is EPSG:4326: sample with lon/lat, not with the scene's UTM metres
                    vals = ki.worldcover_at([(lon[i], lat[i]) for i in idxs], path)
                    wc_values[idxs] = vals
            semantic = ki.semantic_area(wc_values) if not args.no_ki else [''] * len(records)

            fac_pts = shapely.points(world)
            for i, (pol, obj) in enumerate(records):
                pts = obj.get('pts') or []
                if len(pts) < 3:
                    continue
                scene_pts = obj.get('source_pts') or pts
                cx = sum(p[0] for p in scene_pts) / len(scene_pts)
                cy = sum(p[1] for p in scene_pts) / len(scene_pts)
                area = abs(sum(scene_pts[j][0] * scene_pts[(j + 1) % len(scene_pts)][1]
                               - scene_pts[(j + 1) % len(scene_pts)][0] * scene_pts[j][1]
                               for j in range(len(scene_pts)))) / 2.0
                sides = [math.dist(scene_pts[j], scene_pts[(j + 1) % len(scene_pts)]) for j in range(len(scene_pts))]
                kind, fdist, fid, felem = '', '', '', ''
                if geo and geo['facility']['tree'] is not None:
                    geoms_f, kinds_f = geo['facility']['geoms'], geo['facility']['kinds']
                    ids_f, elems_f = geo['facility']['ids'], geo['facility']['elems']
                    hit = geo['facility']['tree'].query(fac_pts[i].buffer(FACILITY_BUFFER_M))
                    if len(hit):
                        d = [fac_pts[i].distance(geoms_f[j]) for j in hit]
                        j = hit[int(min(range(len(hit)), key=lambda t: d[t]))]
                        kind = kinds_f[j]
                        fid = ids_f[j]
                        felem = elems_f[j]
                        fdist = round(fac_pts[i].distance(geoms_f[j]), 1)
                fac_ts = way_dates.get((felem or 'way') + ':' + fid, way_dates.get(fid, '')) if fid else ''
                scene_day = (key.get('start_utc') or '')[:10]
                mmsi = (obj.get('matched_mmsi') or obj.get('candidate_mmsi') or '')
                fine, fine_level, fine_conf, fine_src = classes.get(str(mmsi), ('', '', '', ''))
                signals = dict(
                    ais_unique_match=obj.get('match_status') == 'unique_spatial_candidate',
                    fine_class_resolved=(fine_level == 'fine' and not fine.endswith('_coarse')),
                    valid_area_ok=(obj.get('pixel_status') or '') not in ('', 'zero_only', 'invalid'),
                    coast_known=(obj.get('coast_status') or '') not in ('', 'unavailable', 'missing'),
                    facility_context=bool(kind),
                    facility_map_available=geo is not None,
                )
                row = dict(
                    object_id='%s|%s|%s|%s' % (product, pol, (obj.get('roi_origin') or [0, 0])[0], obj.get('detection_id', '')),
                    product_id=product, port=port, polarization=pol, start_utc=key.get('start_utc', ''),
                    detection_id=obj.get('detection_id', ''), roi_origin=json.dumps(obj.get('roi_origin', [])),
                    conf=obj.get('conf', ''), centroid_x=round(cx, 2), centroid_y=round(cy, 2),
                    world_x=round(world[i][0], 2), world_y=round(world[i][1], 2),
                    area_px2=round(area, 1), obb_long_px=round(max(sides), 1), obb_short_px=round(min(sides), 1),
                    obb_long_deg=round(float(obb_long[i]), 2) if not math.isnan(obb_long[i]) else '',
                    screen_status=obj.get('screen_status', ''), match_status=obj.get('match_status', ''),
                    candidate_mmsi=obj.get('candidate_mmsi', ''), candidate_distance_m=obj.get('candidate_distance_m', ''),
                    matched_mmsi=mmsi, prelabel_class=obj.get('prelabel_class', ''),
                    ais_final_class=fine, ais_class_level=fine_level, ais_class_confidence=fine_conf,
                    ais_class_source=fine_src,
                    fine_class=fine, fine_class_source=fine_src,
                    pixel_status=obj.get('pixel_status', ''), valid_fraction=obj.get('valid_fraction', ''),
                    zero_fraction=obj.get('zero_fraction', ''), coast_status=obj.get('coast_status', ''),
                    coast_exclusion_m=obj.get('coast_exclusion_m', ''),
                    facility_kind=kind, facility_distance_m=fdist, facility_osm_id=fid,
                    facility_osm_element=felem,
                    facility_edit_ts=fac_ts,
                    facility_temporal_valid=('' if not fac_ts else int(fac_ts[:10] <= scene_day)),
                    d_coast_m=round(float(min(dist['coast'][i], dist['coast_derived'][i])), 1), d_quay_m=round(float(dist['quay'][i]), 1),
                    distance_source=('osm_coastline' if dist['coast'][i] < ki.RADIUS
                                     else 'worldcover_derived' if dist['coast_derived'][i] < ki.RADIUS
                                     else 'pipeline_worldcover' if (obj.get('coast_exclusion_m') not in (None, '', 'None'))
                                     else 'unavailable'),
                    d_coast_pipeline_m=obj.get('coast_exclusion_m', ''),
                    d_anchorage_m=round(float(dist['anchorage'][i]), 1), d_fairway_m=round(float(dist['nav'][i]), 1),
                    on_fairway=int(dist['nav'][i] <= ki.ON_FAIRWAY_M),
                    channel_angle_deg='' if (args.no_ki or math.isnan(angle[i])) else round(float(angle[i]), 2),
                    channel_angle_source=angle_source[i],
                    local_ships_500m=int(density.get('local_ships_500m', [0] * len(records))[i]),
                    local_ships_1km=int(density.get('local_ships_1km', [0] * len(records))[i]),
                    nn_distance_m='' if (args.no_ki or i >= len(density.get('nn_distance_m', []))) or math.isnan(density['nn_distance_m'][i]) else round(float(density['nn_distance_m'][i]), 1),
                    heading_consistency_deg='' if (args.no_ki or math.isnan(heading[i])) else round(float(heading[i]), 2),
                    worldcover_class=int(wc_values[i]), semantic_area=semantic[i],
                    human_reviewed=obj.get('human_reviewed', ''), final_dataset_eligible=obj.get('final_dataset_eligible', ''),
                    support_i=sum(1 for v in signals.values() if v),
                    support_missing=';'.join(k for k, v in signals.items() if not v),
                    **{k: int(v) for k, v in signals.items()})
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                rows_written += 1
                stats['objects'] += 1
                by_pc[(port, fine or 'untyped')] += 1
                if kind:
                    stats['facility_hits'] += 1
                if dist['nav'][i] <= ki.ON_FAIRWAY_M:
                    stats['on_fairway'] += 1
            if pi % 10 == 0:
                print('%d/%d products, %d objects, %.0fs' % (pi, len(product_keys), rows_written, time.time() - T0), flush=True)

    with (OUT / 'objects_by_port_class.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['port', 'fine_class', 'objects'])
        for (port, cls), n in sorted(by_pc.items()):
            w.writerow([port, cls, n])
    summary = dict(products=len(product_keys), objects=rows_written,
                   generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   facility_hits=stats['facility_hits'], on_fairway=stats['on_fairway'],
                   signals=SIGNALS, counters=dict(stats),
                   note='facility/navigation geometry is current-snapshot OSM; a hit is review-pending')
    (OUT / 'objects_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

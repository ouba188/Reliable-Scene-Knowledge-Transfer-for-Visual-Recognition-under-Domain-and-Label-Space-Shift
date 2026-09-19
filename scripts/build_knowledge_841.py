"""Build the 841-product knowledge coverage table (read-only over source assets).

Outputs into E:/临时会话/knowledge_set_841/:
  knowledge_coverage_841.csv   841 rows x per-layer knowledge columns
  knowledge_gaps.json          per-layer gap lists (what is still missing)
  knowledge_summary.json       layer completion counts

AIS txt column layout follows E:/Hermes/paper_tracking/extract_port_ais_2024_2025.py:
  [0] epoch [1] mmsi [6] lon [7] lat [10] name [11] type [12] imo [13] callsign [14..17] dims/source
"""
import csv, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path(r'E:/临时会话/safe841_redownload_plan_20260918')
BUNDLE = PLAN / 'annotation_bundle' / 'assets'
PRODUCTS = BUNDLE / 'products'
MAPPING = BUNDLE / 'mapping.csv'
WORLDCOVER_ROOTS = [BUNDLE / 'worldcover', Path(r'E:/临时会话/knowledge_set_841/worldcover')]
INVENTORY = PLAN / 'product_inventory_841.csv'
OUT = Path(r'E:/临时会话/knowledge_set_841')
OSM_DIR = OUT / 'osm'
FILL = OUT / 'mmsi' / 'mmsi_fine_class_filled.csv'

MMSI_RE = re.compile(r'\d{9}')
CLEAN = lambda v: re.sub(r'\s+', ' ', (v or '').strip())


def canon_port(name):
    """AIS meta files carry 7 underscore/space variants; fold them onto the 24 canonical ports."""
    p = re.sub(r'\s+', ' ', (name or '').replace('_', ' ')).strip()
    return {'NewYork': 'New York'}.get(p, p)



def load_mapping():
    """mmsi -> (mapping_status, final_class). Only two columns are needed downstream."""
    table = {}
    with MAPPING.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            table[row['mmsi']] = (row.get('mapping_status', ''), row.get('final_class', ''))
    return table


def load_inventory():
    with INVENTORY.open('r', encoding='utf-8-sig', newline='') as f:
        return {r['product_id']: r for r in csv.DictReader(f)}


def wc_tiles(min_lon, min_lat, max_lon, max_lat):
    """ESA WorldCover v200 tiles are 3x3 degrees: latitude 2 digits, longitude 3 (N03E102, S33E150)."""
    names = set()
    for lat in range(int(min_lat // 3) * 3, int(max_lat // 3) * 3 + 1, 3):
        for lon in range(int(min_lon // 3) * 3, int(max_lon // 3) * 3 + 1, 3):
            names.add('ESA_WorldCover_10m_2021_v200_%s%02d%s%03d_Map.tif' % (
                'N' if lat >= 0 else 'S', abs(lat), 'E' if lon >= 0 else 'W', abs(lon)))
    return names


def scan_ais(product_dir):
    """Return AIS-layer stats for one product. Streams; no full-file buffering."""
    rows = bad_fields = bad_mmsi = 0
    mmsi_rows = Counter()
    ts = []
    windows = set()
    for txt in sorted((product_dir / 'ais').glob('*.txt')):
        with txt.open('r', encoding='utf-8', errors='replace', newline='') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                cols = line.split(',')
                if len(cols) < 18:
                    bad_fields += 1
                    continue
                mmsi = CLEAN(cols[1])
                if not (MMSI_RE.fullmatch(mmsi) and mmsi != '000000000'):
                    bad_mmsi += 1
                    continue
                mmsi_rows[mmsi] += 1
                try:
                    ts.append(int(cols[0]))
                except ValueError:
                    pass
    return rows, bad_fields, bad_mmsi, mmsi_rows, ts


def load_fill():
    """MMSI -> (final_class, source) additionally resolved by the fill pipeline, if it ran."""
    if not FILL.is_file():
        return {}
    with FILL.open('r', encoding='utf-8-sig', newline='') as f:
        return {r['mmsi']: (r['final_class'], r['source']) for r in csv.DictReader(f) if r['final_class']}


def load_wc_audit():
    """Authoritative per-product tile needs from audit_worldcover_tiles.py (annotation footprint + 1 km)."""
    path = OUT / 'worldcover_per_product.csv'
    if not path.is_file():
        return {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return {r['product_id']: dict(needed=int(r['needed'] or 0), have=int(r['have'] or 0),
                                      missing=r['missing']) for r in csv.DictReader(f)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping()
    fill = load_fill()
    wc_audit = load_wc_audit()
    wc_audit_json = json.loads((OUT / 'worldcover_audit.json').read_text(encoding='utf-8')) \
        if (OUT / 'worldcover_audit.json').is_file() else {}
    WC_UNPUBLISHED = set(wc_audit_json.get('unavailable_in_bucket') or [])
    inventory = load_inventory()
    product_ids = sorted(p.name for p in PRODUCTS.iterdir() if p.is_dir())
    have_wc = {p.name for root in WORLDCOVER_ROOTS if root.is_dir() for p in root.glob('*.tif')}
    osm_present = {p.stem for p in OSM_DIR.glob('*.geojson')} if OSM_DIR.is_dir() else set()
    osm_present |= {p.stem for p in OSM_DIR.glob('*.json') if not p.name.endswith('.provenance.json')}

    wc_missing_all = Counter()
    rows_out = []
    gaps = defaultdict(list)
    fine_class_products = defaultdict(set)
    unresolved = defaultdict(lambda: {'rows': 0, 'status': Counter(), 'ports': set()})
    global_mmsi = set()

    for n, pid in enumerate(product_ids, 1):
        pdir = PRODUCTS / pid
        meta = {}
        for mj in sorted((pdir / 'ais').glob('*.meta.json')):
            try:
                meta = json.loads(mj.read_text(encoding='utf-8'))
                break
            except ValueError:
                continue
        bbox = meta.get('bbox') or {}
        port = canon_port(meta.get('primary_port') or (inventory.get(pid, {}).get('port') or ''))
        ais_files = sorted((pdir / 'ais').glob('*.txt'))
        rows, bad_fields, bad_mmsi, mmsi_rows, ts = scan_ais(pdir)
        global_mmsi |= set(mmsi_rows)

        qs, qe = meta.get('query_start_epoch'), meta.get('query_end_epoch')

        # fine-grained ship type join: curated mapping layer first, then the fill pipeline's evidence
        resolved = resolved_fill = resolved_coarse = missing = conflict = unknown = 0
        for mmsi in mmsi_rows:
            status, final_class = mapping.get(mmsi, ('', ''))
            if final_class and status == 'fine_class_mapped':
                resolved += 1
                fine_class_products[final_class].add(pid)
                continue
            filled_class, _filled_source = fill.get(mmsi, ('', ''))
            if filled_class:
                resolved += 1
                resolved_fill += 1
                fine_class_products[filled_class].add(pid)
                continue
            if final_class:
                # coarse-only label (cargo_coarse / tanker_coarse ...): usable as knowledge, not a fine class
                resolved_coarse += 1
                fine_class_products[final_class].add(pid)
                continue
            if not status:
                unknown += 1
                key = 'not_in_mapping'
            elif 'conflict' in status:
                conflict += 1
                key = status
            else:
                missing += 1
                key = status
            unresolved[mmsi]['rows'] += mmsi_rows[mmsi]
            unresolved[mmsi]['status'][key] += 1
            unresolved[mmsi]['ports'].add(port)

        need_wc, lack_wc = set(), set()
        audit = wc_audit.get(pid)
        if audit:
            need_wc, have_n = set(), audit['have']
            lack_wc = {t for t in audit['missing'].split(';') if t}
            wc_ok = bool(audit['missing']) is False
        elif bbox:
            need_wc = wc_tiles(bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat'])
            lack_wc = need_wc - have_wc
            have_n = len(need_wc - lack_wc)
            wc_ok = not lack_wc
        else:
            have_n, wc_ok = 0, False
        for t in lack_wc:
            wc_missing_all[t] += 1

        inv = inventory.get(pid, {})
        local_ok = inv.get('local_complete') == 'True'
        remote_ok = inv.get('remote_22559_complete') == 'True'
        has_osm = port in osm_present

        blockers = []
        if not (pdir / 'iw-vv.xml').is_file() or not (pdir / 'iw-vh.xml').is_file():
            blockers.append('sentinel1_xml')
        if rows == 0:
            blockers.append('no_ais_rows')
        if unknown:
            blockers.append('mmsi_not_in_mapping')
        if missing or conflict:
            blockers.append('fine_class_unresolved')
        if lack_wc:
            blockers.append('worldcover_tile_esa_unpublished'
                            if all(t in WC_UNPUBLISHED for t in lack_wc) else 'worldcover_tile')
        if not has_osm:
            blockers.append('osm_facility_map')
        if not local_ok:
            blockers.append('local_raster' if not remote_ok else 'local_raster_remote_only')

        rows_out.append(dict(
            product_id=pid, port=port, track=meta.get('product_track', inv.get('track', '')),
            start_utc=meta.get('acquisition_start_utc', inv.get('start_time', '')),
            bbox_min_lon=bbox.get('min_lon', ''), bbox_min_lat=bbox.get('min_lat', ''),
            bbox_max_lon=bbox.get('max_lon', ''), bbox_max_lat=bbox.get('max_lat', ''),
            has_iw_vv=(pdir / 'iw-vv.xml').is_file(), has_iw_vh=(pdir / 'iw-vh.xml').is_file(),
            ais_files=len(ais_files), ais_rows=rows, ais_rows_bad_fields=bad_fields,
            ais_rows_bad_mmsi=bad_mmsi, ais_mmsi_unique=len(mmsi_rows),
            ais_ts_min=min(ts) if ts else '', ais_ts_max=max(ts) if ts else '',
            ais_query_start_epoch=qs or '', ais_query_end_epoch=qe or '',
            fine_resolved=resolved, fine_resolved_from_fill=resolved_fill,
            fine_resolved_coarse=resolved_coarse,
            fine_missing=missing, fine_conflict=conflict, fine_unknown=unknown,
            fine_classes=';'.join(sorted({(mapping.get(m, ('', ''))[1] or fill.get(m, ('', ''))[0])
                                          for m in mmsi_rows} - {''})),
            worldcover_need=len(need_wc) or have_n, worldcover_have=have_n,
            worldcover_ok=wc_ok,
            worldcover_missing=';'.join(sorted(lack_wc)),
            worldcover_gap_esa_unpublished=all(t in WC_UNPUBLISHED for t in lack_wc) if lack_wc else False,
            osm_facility_map=has_osm,
            raster_local_complete=local_ok, raster_remote_22559_complete=remote_ok,
            raster_action=inv.get('action', ''),
            detection_status='pending_rasters',
            knowledge_ready=not blockers,
            blockers=';'.join(blockers),
        ))

        if rows == 0:
            gaps['no_ais_rows'].append(pid)
        if lack_wc:
            gaps['worldcover_tile_missing'].append(pid)
        if not local_ok and not remote_ok:
            gaps['no_raster_anywhere'].append(pid)
        if n % 50 == 0:
            print('scanned %d/%d (%.0fs)' % (n, len(product_ids), time.time() - t0), flush=True)

    fields = list(rows_out[0].keys())
    with (OUT / 'knowledge_coverage_841.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    gaps['mmsi_not_in_mapping'] = [
        dict(mmsi=m, rows=v['rows'], statuses=dict(v['status']), ports=sorted(v['ports']))
        for m, v in sorted(unresolved.items(), key=lambda kv: -kv[1]['rows'])]
    gaps['worldcover_tiles_missing'] = sorted(wc_missing_all)
    gaps['ports_without_osm_map'] = sorted({r['port'] for r in rows_out if r['port'] and not r['osm_facility_map']})
    (OUT / 'knowledge_gaps.json').write_text(json.dumps(gaps, ensure_ascii=False, indent=1), encoding='utf-8')

    summary = dict(
        built_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), products=len(rows_out),
        layer_prechecks=dict(
            sentinel1_xml=sum(1 for r in rows_out if r['has_iw_vv'] and r['has_iw_vh']),
            ais_files=sum(1 for r in rows_out if r['ais_files'] > 0),
            ais_rows=sum(1 for r in rows_out if r['ais_rows'] > 0),
            fine_class_all_resolved=sum(1 for r in rows_out if r['fine_missing'] == 0 and r['fine_conflict'] == 0 and r['fine_unknown'] == 0 and r['ais_mmsi_unique'] > 0),
            worldcover_complete=sum(1 for r in rows_out if r['worldcover_ok']),
            worldcover_gap_only_esa_unpublished=sum(1 for r in rows_out if not r['worldcover_ok'] and r['worldcover_gap_esa_unpublished']),
            osm_facility_map=sum(1 for r in rows_out if r['osm_facility_map']),
            raster_local_complete=sum(1 for r in rows_out if r['raster_local_complete']),
            raster_remote_only=sum(1 for r in rows_out if not r['raster_local_complete'] and r['raster_remote_22559_complete']),
            raster_missing_both=sum(1 for r in rows_out if not r['raster_local_complete'] and not r['raster_remote_22559_complete']),
            fully_ready=sum(1 for r in rows_out if r['knowledge_ready']),
        ),
        ais=dict(rows=sum(r['ais_rows'] for r in rows_out),
                 unique_mmsi=len(global_mmsi),
                 bad_field_rows=sum(r['ais_rows_bad_fields'] for r in rows_out),
                 bad_mmsi_rows=sum(r['ais_rows_bad_mmsi'] for r in rows_out)),
        fine_class=dict(resolved_pairs=sum(r['fine_resolved'] for r in rows_out),
                        resolved_from_fill_pairs=sum(r['fine_resolved_from_fill'] for r in rows_out),
                        coarse_only_pairs=sum(r['fine_resolved_coarse'] for r in rows_out),
                        missing_pairs=sum(r['fine_missing'] for r in rows_out),
                        conflict_pairs=sum(r['fine_conflict'] for r in rows_out),
                        unknown_pairs=sum(r['fine_unknown'] for r in rows_out),
                        distinct_unresolved_mmsi=len(unresolved)),
        worldcover=dict(present_tiles=len(have_wc), missing_tiles=sorted(wc_missing_all),
                        products_affected=len(gaps['worldcover_tile_missing'])),
        fine_classes_seen={k: len(v) for k, v in sorted(fine_class_products.items())},
        elapsed_s=round(time.time() - t0, 1),
    )
    (OUT / 'knowledge_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')

    assert len(rows_out) == 841, 'expected 841 products, got %d' % len(rows_out)
    print(json.dumps(summary['layer_prechecks'], ensure_ascii=False, indent=1))
    print('elapsed %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()

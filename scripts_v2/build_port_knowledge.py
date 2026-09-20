"""Port-level knowledge vector d_p = (meta, facility, function, scene) for the 24 ports.

Sources, all local:
  meta      keys/products.csv (scene dates, orbit, pass) per product
  facility  facility geometry kinds per port (local 2026-07 layers + backfilled extracts)
  function  facility-mix proxies (tanker / container / bulk / ferry / shipyard / fishing)
  scene     AIS-derived ship-group statistics (class shares, size distribution, density)

Writes knowledge_set_841/ports/port_knowledge.csv (+ .json) with one row per port.
Class labels come from the curated mapping layer first, then the MMSI fill pipeline.
"""
import csv, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path

KS = Path(r'E:/临时会话/knowledge_set_841')
BUNDLE = Path(r'E:/临时会话/safe841_redownload_plan_20260918/annotation_bundle')
PRODUCTS = BUNDLE / 'assets' / 'products'
KEYS = KS / 'keys' / 'products.csv'
MAPPING = BUNDLE / 'assets' / 'mapping.csv'
FILL = KS / 'mmsi' / 'mmsi_fine_class_filled.csv'
OUT = KS / 'ports'

FUNCTION_PROXIES = {
    'tanker_terminals': ('storage_tank', 'infra_oil_tanks', 'crane'),
    'container_terminals': ('infra_cranes', 'crane', 'container', 'infra_container'),
    'bulk_terminals': ('silo', 'infra_silos_conveyors', 'conveyor'),
    'ferry_terminals': ('ferry', 'passenger'),
    'shipyard': ('dry_dock', 'shipyard', 'slipway'),
    'rail_logistics': ('infra_rail_logistics', 'railway'),
    'energy_fuel': ('infra_energy_fuel', 'pipeline', 'infra_pipelines'),
}


def load_classes():
    table = {}
    with MAPPING.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if (r.get('mapping_status') or '') == 'fine_class_mapped' and r.get('final_class'):
                table[r['mmsi']] = r['final_class']
    if FILL.is_file():
        with FILL.open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r['final_class'] and r['mmsi'] not in table:
                    table[r['mmsi']] = r['final_class']
    return table


def facility_counts(port):
    import build_object_table_local as B
    geo = B.load_port_geometry(port)
    if geo is None or not len(geo):
        return Counter(), 0
    kinds = geo['facility_kind'].astype(str)
    return Counter(kinds), len(geo)


def scene_stats(port, product_ids, classes):
    """AIS-derived ship group statistics for one port (the ξ_t term of the scene state)."""
    rows = 0
    mmsi_rows = Counter()
    class_rows = Counter()
    lengths, beams = [], []
    lon_min = lat_min = 1e9
    lon_max = lat_max = -1e9
    for pid in product_ids:
        for txt in sorted((PRODUCTS / pid / 'ais').glob('*.txt')):
            with txt.open('r', encoding='utf-8', errors='replace', newline='') as f:
                for line in f:
                    cols = line.rstrip('\n').split(',')
                    if len(cols) < 18:
                        continue
                    mmsi = cols[1].strip()
                    if not (re.fullmatch(r'\d{9}', mmsi) and mmsi != '000000000'):
                        continue
                    rows += 1
                    mmsi_rows[mmsi] += 1
                    cls = classes.get(mmsi)
                    if cls:
                        class_rows[cls] += 1
                    for raw, sink in ((cols[14], lengths), (cols[15], beams)):
                        try:
                            v = float(raw)
                            if 0 < v < 500:
                                sink.append(v)
                        except ValueError:
                            pass
                    try:
                        lon, lat = float(cols[6]), float(cols[7])
                        lon_min, lon_max = min(lon_min, lon), max(lon_max, lon)
                        lat_min, lat_max = min(lat_min, lat), max(lat_max, lat)
                    except ValueError:
                        pass
    top = class_rows.most_common()
    total_classed = sum(class_rows.values()) or 1
    return dict(ais_rows=rows, distinct_mmsi=len(mmsi_rows),
                classed_rows=total_classed,
                class_share_json=json.dumps({k: round(v / total_classed, 4) for k, v in top[:12]}),
                top_class=top[0][0] if top else '',
                mean_length_m=round(sum(lengths) / len(lengths), 1) if lengths else '',
                mean_beam_m=round(sum(beams) / len(beams), 1) if beams else '',
                bbox_lon_min=round(lon_min, 3) if lon_min < 1e8 else '',
                bbox_lon_max=round(lon_max, 3) if lon_max > -1e8 else '',
                bbox_lat_min=round(lat_min, 3) if lat_min < 1e8 else '',
                bbox_lat_max=round(lat_max, 3) if lat_max > -1e8 else '')


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    classes = load_classes()
    by_port = defaultdict(list)
    meta = defaultdict(list)
    with KEYS.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            by_port[r['port']].append(r['product_id'])
            meta[r['port']].append(r)

    rows = []
    for port in sorted(by_port):
        kinds, total_geo = facility_counts(port)
        scenes = sorted({m['start_utc'][:10] for m in meta[port] if m.get('start_utc')})
        orbits = Counter(m['relative_orbit'] for m in meta[port] if m.get('relative_orbit'))
        hours = Counter(m['start_utc'][11:13] for m in meta[port] if m.get('start_utc'))
        row = dict(
            port=port, products=len(by_port[port]), facility_features=total_geo,
            scene_dates=len(scenes), first_scene=scenes[0] if scenes else '', last_scene=scenes[-1] if scenes else '',
            orbit_mode=orbits.most_common(1)[0][0] if orbits else '',
            acquisition_hour_mode=hours.most_common(1)[0][0] + ':00' if hours else '',
        )
        for kind, n in kinds.most_common(14):
            row['fac_' + re.sub(r'\W+', '_', kind)[:32]] = n
        for name, keys in FUNCTION_PROXIES.items():
            row[name] = sum(v for k, v in kinds.items() if any(t in k for t in keys))
        row.update(scene_stats(port, by_port[port], classes))
        rows.append(row)
        print('%-18s fac=%-6d ais=%-9d mmsi=%-7d top=%s' % (
            port, total_geo, row['ais_rows'], row['distinct_mmsi'], row['top_class']), flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('port', 'products', 'facility_features'), k))
    with (OUT / 'port_knowledge.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / 'port_knowledge.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print('ports=%d columns=%d elapsed=%.0fs' % (len(rows), len(fields), time.time() - t0))


if __name__ == '__main__':
    main()

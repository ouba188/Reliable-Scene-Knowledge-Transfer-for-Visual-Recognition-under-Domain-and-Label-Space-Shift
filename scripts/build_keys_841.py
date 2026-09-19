"""Canonical keys for the 841-product knowledge set: product primary keys + port name aliases.

Step 2 of the research plan: one product_id key, one canonical port name, joinnable everywhere.
Emits keys/products.csv, keys/port_aliases.csv, keys/keys_report.json and asserts 1:1 coverage.
"""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path(r'E:/临时会话/safe841_redownload_plan_20260918')
BUNDLE = PLAN / 'annotation_bundle' / 'assets'
OUT = Path(r'E:/临时会话/knowledge_set_841')
KEYS = OUT / 'keys'
ALIAS = {'NewYork': 'New York'}


def canon_port(name):
    p = re.sub(r'\s+', ' ', (name or '').replace('_', ' ')).strip()
    return ALIAS.get(p, p)


def main():
    KEYS.mkdir(parents=True, exist_ok=True)
    with (PLAN / 'product_inventory_841.csv').open('r', encoding='utf-8-sig', newline='') as f:
        inventory = {r['product_id']: r for r in csv.DictReader(f)}
    with (OUT / 'knowledge_coverage_841.csv').open('r', encoding='utf-8-sig', newline='') as f:
        coverage = {r['product_id']: r for r in csv.DictReader(f)}
    bundle_ids = {p.name for p in (BUNDLE / 'products').iterdir() if p.is_dir()}

    rows, aliases = [], Counter()
    for pid in sorted(coverage):
        cov, inv = coverage[pid], inventory.get(pid, {})
        raw_port = ''
        # recover the raw (pre-canonical) port string from the meta file so aliases stay auditable
        for mj in sorted((BUNDLE / 'products' / pid / 'ais').glob('*.meta.json')):
            try:
                raw_port = json.loads(mj.read_text(encoding='utf-8')).get('primary_port') or raw_port
            except ValueError:
                pass
            break
        if raw_port and raw_port != canon_port(raw_port):
            aliases[raw_port] += 1
        rows.append(dict(
            product_id=pid, port=canon_port(cov.get('port') or inv.get('port')),
            track=cov.get('track') or inv.get('track', ''), relative_orbit=inv.get('relative_orbit', ''),
            start_utc=cov.get('start_utc') or inv.get('start_time', ''), stop_utc=inv.get('stop_time', ''),
            footprint_min_lon=cov.get('bbox_min_lon', ''), footprint_min_lat=cov.get('bbox_min_lat', ''),
            footprint_max_lon=cov.get('bbox_max_lon', ''), footprint_max_lat=cov.get('bbox_max_lat', ''),
            ais_files=cov.get('ais_files', ''), ais_rows=cov.get('ais_rows', ''),
            ais_mmsi_unique=cov.get('ais_mmsi_unique', ''),
            fine_resolved=cov.get('fine_resolved', ''), fine_resolved_from_fill=cov.get('fine_resolved_from_fill', ''),
            fine_resolved_coarse=cov.get('fine_resolved_coarse', ''),
            fine_unresolved=cov.get('fine_missing', ''), fine_conflict=cov.get('fine_conflict', ''),
            worldcover_need=cov.get('worldcover_need', ''), worldcover_have=cov.get('worldcover_have', ''),
            raster_polarizations='VV;VH' if cov.get('has_iw_vv') == 'True' and cov.get('has_iw_vh') == 'True' else '',
            raster_local_complete=cov.get('raster_local_complete', ''),
            raster_remote_22559_complete=cov.get('raster_remote_22559_complete', ''),
            raster_action=cov.get('raster_action', ''),
            detection_status=cov.get('detection_status', ''),
            knowledge_ready=cov.get('knowledge_ready', ''),
            blockers=cov.get('blockers', '')))

    fields = list(rows[0].keys())
    with (KEYS / 'products.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (KEYS / 'port_aliases.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['raw_name', 'canonical_name', 'products'])
        for raw, n in sorted(aliases.items()):
            w.writerow([raw, canon_port(raw), n])

    report = dict(
        products_in_coverage=len(coverage), products_in_inventory=len(inventory), products_in_bundle=len(bundle_ids),
        missing_from_inventory=sorted(set(coverage) - set(inventory)),
        missing_from_bundle=sorted(set(coverage) - bundle_ids),
        ports=len({r['port'] for r in rows}),
        port_aliases=dict(aliases),
        products_per_port=dict(sorted(Counter(r['port'] for r in rows).items())),
        key_columns=['product_id', 'port'], key_notes='product_id is the Sentinel-1 product name (primary key); '
                                                    'port is the canonical 24-port name; track is auxiliary only',
    )
    (KEYS / 'keys_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

    assert len(rows) == 841, 'expected 841 product keys, got %d' % len(rows)
    assert not report['missing_from_inventory'] and not report['missing_from_bundle'], report
    assert len({r['port'] for r in rows}) == 24, 'expected 24 canonical ports'
    print(json.dumps({k: v for k, v in report.items() if k != 'products_per_port'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

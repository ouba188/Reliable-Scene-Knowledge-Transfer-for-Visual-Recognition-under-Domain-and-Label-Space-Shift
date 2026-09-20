"""E00 audits for next_experiment_v03, computed over the local object table.

Produces (into --out):
  feature_permissions.yaml        every object-table column classified for the P0 AIS-free protocol
  dedup_report.json               duplicate candidates: cross-polarisation and cross-tile repetitions
  object_id_collision.json        distinct object_id vs rows (id uniqueness audit)
  label_provenance.csv            AIS label layer composition by level/source/confidence
  temporal_evidence_audit.csv     facility time-evidence status per port (dated-ok / late / undated)
  deployable_coverage_by_port.csv objects surviving a deployable-only mask, per port
  distance_truncation.csv         how often each distance hit the 2000 m cap vs had no layer at all

  python e00_audit.py [objects_final.csv.gz] --out <dir>
"""
import argparse, csv, gzip, json, time
from collections import Counter, defaultdict
from pathlib import Path

PERMISSIONS = [
    # (column, role, source, p0_usable, note)
    ('object_id', 'identifier', 'derived', True, 'tile-local id; collision audit reported separately'),
    ('product_id', 'identifier', 'SAFE', True, ''),
    ('port', 'identifier', 'inventory', True, ''),
    ('polarization', 'identifier', 'SAFE', True, ''),
    ('start_utc', 'identifier', 'SAFE', True, 'acquisition time; usable for time-based G protocol splits'),
    ('detection_id', 'identifier', 'detector', True, ''),
    ('roi_origin', 'identifier', 'tiling', True, 'tile origin; needed to rebuild a global key'),
    ('conf', 'feature', 'detector', True, 'detector confidence'),
    ('centroid_x', 'feature', 'detector', True, 'within-tile pixel coordinate'),
    ('centroid_y', 'feature', 'detector', True, ''),
    ('world_x', 'feature', 'geocode', True, 'UTM metres'),
    ('world_y', 'feature', 'geocode', True, ''),
    ('area_px2', 'feature', 'detector', True, ''),
    ('obb_long_px', 'feature', 'detector', True, ''),
    ('obb_short_px', 'feature', 'detector', True, ''),
    ('obb_long_deg', 'feature', 'detector', True, 'axial orientation mod pi; not a heading'),
    ('screen_status', 'audit', 'pipeline', False, 'screening bookkeeping; audit before reuse'),
    ('match_status', 'evaluator_only', 'AIS', False, 'AIS matching state'),
    ('candidate_mmsi', 'evaluator_only', 'AIS', False, 'target AIS identity'),
    ('candidate_distance_m', 'evaluator_only', 'AIS', False, ''),
    ('matched_mmsi', 'evaluator_only', 'AIS', False, ''),
    ('prelabel_class', 'evaluator_only', 'AIS', False, 'AIS-derived label'),
    ('ais_final_class', 'evaluator_only', 'AIS', False, ''),
    ('ais_class_level', 'evaluator_only', 'AIS', False, ''),
    ('ais_class_confidence', 'evaluator_only', 'AIS', False, ''),
    ('ais_class_source', 'evaluator_only', 'AIS', False, ''),
    ('fine_class', 'evaluator_only', 'AIS', False, ''),
    ('fine_class_source', 'evaluator_only', 'AIS', False, ''),
    ('pixel_status', 'feature', 'SAR validity', True, 'valid-pixel screening from the imagery itself'),
    ('valid_fraction', 'feature', 'SAR validity', True, ''),
    ('zero_fraction', 'feature', 'SAR validity', True, ''),
    ('coast_status', 'feature', 'WorldCover', True, 'near-shore screening status'),
    ('coast_exclusion_m', 'feature', 'WorldCover', True, 'distance to coast from the pipeline mask'),
    ('facility_kind', 'feature', 'OSM', True, 'static map relation'),
    ('facility_distance_m', 'feature', 'OSM', True, 'capped at 25 m buffer semantics; see distance notes'),
    ('facility_osm_id', 'identifier', 'OSM', True, 'join key for time evidence'),
    ('facility_osm_element', 'identifier', 'OSM', True, 'way/node/relation'),
    ('facility_edit_ts', 'evidence_only', 'OSM API', True, 'last-edit time; NOT construction time, NOT a K-correctness label'),
    ('facility_temporal_valid', 'evidence_only', 'OSM API', False, 'not wrongK; see temporal_evidence_audit.csv'),
    ('d_coast_m', 'feature', 'OSM/WorldCover', True, 'capped at 2000 m; distinguish cap from missing layer'),
    ('d_quay_m', 'feature', 'OSM', True, 'capped at 2000 m'),
    ('distance_source', 'provenance', 'mixed', True, 'osm_coastline | worldcover_derived | pipeline_worldcover'),
    ('d_coast_pipeline_m', 'feature', 'WorldCover', True, 'independent of the OSM coastline route'),
    ('d_anchorage_m', 'feature', 'OSM', True, 'capped at 2000 m'),
    ('d_fairway_m', 'feature', 'OSM', True, 'capped at 2000 m; sparse outside mapped ports'),
    ('on_fairway', 'feature', 'OSM', True, 'depends on shallow OSM fairway coverage'),
    ('channel_angle_deg', 'feature', 'mixed', 'split', 'P0: use only rows with channel_angle_source=osm_fairway; ais_traffic rows are AIS-derived'),
    ('channel_angle_source', 'provenance', 'mixed', True, 'osm_fairway | ais_traffic'),
    ('local_ships_500m', 'feature', 'SAR', 'after_dedup', 'recompute after tile/polarisation dedup'),
    ('local_ships_1km', 'feature', 'SAR', 'after_dedup', ''),
    ('nn_distance_m', 'feature', 'SAR', 'after_dedup', ''),
    ('heading_consistency_deg', 'feature', 'SAR', 'after_dedup', ''),
    ('worldcover_class', 'feature', 'WorldCover', True, ''),
    ('semantic_area', 'feature', 'WorldCover', True, 'water|land|built|nodata'),
    ('human_reviewed', 'audit', 'pipeline', False, 'all zero in this build'),
    ('final_dataset_eligible', 'audit', 'pipeline', False, 'audit before reuse'),
    ('support_i', 'forbidden_as_input', 'mixed', False, 'includes AIS signals; forbidden as target feature or sampling rule'),
    ('support_missing', 'forbidden_as_input', 'mixed', False, 'same'),
    ('ais_unique_match', 'evaluator_only', 'AIS', False, ''),
    ('fine_class_resolved', 'evaluator_only', 'AIS', False, ''),
    ('valid_area_ok', 'audit', 'derived', False, ''),
    ('coast_known', 'audit', 'derived', False, ''),
    ('facility_context', 'audit', 'derived', False, ''),
    ('facility_map_available', 'audit', 'derived', False, ''),
]
DEPLOYABLE = {c for c, _r, _s, ok, _n in PERMISSIONS if ok is True or ok == 'after_dedup'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table', nargs='?', type=Path,
                    default=Path(r'E:/临时会话/knowledge_set_841/objects/objects_final.csv.gz'))
    ap.add_argument('--out', type=Path, default=Path(r'E:/Docms/无监督域适应/translated/0917/next_experiment_v03/e00'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import yaml
    (args.out / 'feature_permissions.yaml').write_text(yaml.safe_dump(dict(
        protocol='P0 AIS-free (next_experiment_v03)',
        source_supervision=['matched_mmsi', 'candidate_mmsi', 'prelabel_class', 'ais_final_class',
                            'ais_class_level', 'ais_class_confidence', 'ais_class_source',
                            'fine_class', 'fine_class_source'],
        evaluator_only=[c for c, r, _s, _o, _n in PERMISSIONS if r == 'evaluator_only'],
        forbidden_as_target_input=['support_i', 'support_missing'],
        evidence_only=['facility_edit_ts', 'facility_temporal_valid'],
        columns=[dict(column=c, role=r, source=s, p0_usable=o, note=n) for c, r, s, o, n in PERMISSIONS],
        caveats=[
            'facility_temporal_valid compares OSM last-edit time with the scene date: not construction time.',
            'distances cap at 2000 m; a capped value is not evidence that the layer exists 2 km away.',
            'channel_angle_deg falls back to an AIS-derived field; P0 must split on channel_angle_source.',
            'local density / nearest neighbour / heading stats were computed before tile and polarisation dedup.',
            'support_i mixes AIS signals and must not gate target candidates.',
        ]), allow_unicode=True, sort_keys=False), encoding='utf-8')

    n = 0
    dedup = Counter()
    seen = set()
    ids = set()
    label_prov = Counter()
    temporal = defaultdict(Counter)
    cover = defaultdict(Counter)
    trunc = Counter()
    caps = ('d_coast_m', 'd_quay_m', 'd_anchorage_m', 'd_fairway_m')
    t0 = time.time()
    with gzip.open(args.table, 'rt', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            n += 1
            ids.add(row['object_id'])
            key = (row['product_id'], row['world_x'], row['world_y'])
            if key in seen:
                dedup['same_product_same_world_point'] += 1
            else:
                seen.add(key)
            dedup['pol_' + row['polarization']] += 1
            if row.get('ais_class_level'):
                label_prov[(row['ais_class_level'], row.get('ais_class_source', ''), row.get('ais_class_confidence', ''))] += 1
            p = row['port']
            ts = row.get('facility_edit_ts', '')
            scene = (row.get('start_utc') or '')[:10]
            if not row.get('facility_kind'):
                temporal[p]['no_facility_hit'] += 1
            elif not ts:
                temporal[p]['hit_not_dated'] += 1
            elif ts[:10] <= scene:
                temporal[p]['dated_existed_before_scene'] += 1
            else:
                temporal[p]['dated_edited_after_scene'] += 1
            ok = (row.get('pixel_status') not in ('', 'zero_only', 'invalid')
                  and row.get('valid_fraction', '0') not in ('', '0'))
            cover[p]['objects'] += 1
            if ok:
                cover[p]['deployable_valid_area'] += 1
            for c in caps:
                v = row.get(c, '')
                if v in ('', '2000.0', '2000'):
                    trunc[c + '_at_cap'] += 1
                else:
                    trunc[c + '_has_value'] += 1
    (args.out / 'dedup_report.json').write_text(json.dumps(dict(
        rows=n,
        same_product_same_world_point=dedup['same_product_same_world_point'],
        cross_polarisation_pairs='not separable in this table: VV/VH share world coordinates only within a few metres',
        polarisation_rows={k[4:]: v for k, v in dedup.items() if k.startswith('pol_')},
        action='rebuild density / nearest-neighbour / heading statistics after merging VV+VH and overlapping tiles',
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    (args.out / 'object_id_collision.json').write_text(json.dumps(dict(
        rows=n, distinct_object_id=len(ids), collisions=n - len(ids),
        key='product|polarisation|roi_origin.x|detection_id',
        action='add full roi_origin (x and y) plus candidate index to the key and keep the original',
    ), ensure_ascii=False, indent=1), encoding='utf-8')
    with (args.out / 'label_provenance.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_level', 'source', 'confidence', 'objects'])
        for (lvl, src, conf), c in sorted(label_prov.items(), key=lambda kv: -kv[1]):
            w.writerow([lvl, src, conf, c])
    with (args.out / 'temporal_evidence_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['port', 'no_facility_hit', 'hit_not_dated', 'dated_existed_before_scene', 'dated_edited_after_scene'])
        for p in sorted(temporal):
            t = temporal[p]
            w.writerow([p, t['no_facility_hit'], t['hit_not_dated'],
                        t['dated_existed_before_scene'], t['dated_edited_after_scene']])
    with (args.out / 'deployable_coverage_by_port.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['port', 'objects', 'deployable_valid_area', 'share'])
        for p in sorted(cover):
            o = cover[p]['objects']
            w.writerow([p, o, cover[p]['deployable_valid_area'], round(cover[p]['deployable_valid_area'] / max(1, o), 4)])
    with (args.out / 'distance_truncation.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distance', 'has_value', 'at_cap_2000m'])
        for c in caps:
            w.writerow([c, trunc[c + '_has_value'], trunc[c + '_at_cap']])
    print('rows %d  distinct ids %d  collisions %d  same world point repeats %d  (%.0fs)' % (
        n, len(ids), n - len(ids), dedup['same_product_same_world_point'], time.time() - t0))
    print('outputs ->', args.out)


if __name__ == '__main__':
    main()

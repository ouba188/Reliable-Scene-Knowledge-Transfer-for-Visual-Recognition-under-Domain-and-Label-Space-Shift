"""Final verification pass over the object table: coverage, dating, support, class levels."""
import csv, gzip, sys, time
from collections import Counter
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else r'E:/临时会话/knowledge_set_841/objects/objects_final.csv.gz')
n = 0
c = Counter()
t0 = time.time()
with gzip.open(p, 'rt', encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        n += 1
        c['facility_hit'] += bool(r['facility_kind'])
        c['feature_id'] += bool(r['facility_osm_id'])
        c['dated'] += bool(r.get('facility_edit_ts'))
        if r.get('facility_temporal_valid') == '1':
            c['temporal_ok'] += 1
        elif r.get('facility_temporal_valid') == '0':
            c['temporal_late'] += 1
        c['support_' + r['support_i']] += 1
        c['level_' + (r.get('ais_class_level') or 'none')] += 1
        c['angle_' + (r.get('channel_angle_source') or 'none')] += 1
        c['dcoast_' + (r.get('distance_source') or 'none')] += 1
        c['on_fairway'] += r.get('on_fairway') == '1'
        c['surface_' + (r.get('semantic_area') or 'none')] += 1
print('对象 %d  用时 %.0fs' % (n, time.time() - t0))
for k in sorted(c):
    print('%-28s %d' % (k, c[k]))

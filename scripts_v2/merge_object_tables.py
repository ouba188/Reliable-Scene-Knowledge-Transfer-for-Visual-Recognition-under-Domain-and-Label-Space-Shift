"""Merge object-table shards and write the combined per-port summary.

  python merge_object_tables.py out/objects_all.csv.gz objects/objects.csv.gz objects/objects_newcastle.csv.gz
"""
import csv, gzip, sys, time
from collections import Counter, defaultdict
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    shards = [Path(p) for p in sys.argv[2:]]
    t0 = time.time()
    per_port = defaultdict(Counter)
    support = defaultdict(Counter)
    writer = None
    total = 0
    with gzip.open(out, 'wt', encoding='utf-8-sig', newline='') as dst:
        for shard in shards:
            with gzip.open(shard, 'rt', encoding='utf-8-sig', newline='') as src:
                for row in csv.DictReader(src):
                    if writer is None:
                        writer = csv.DictWriter(dst, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)
                    total += 1
                    p = row['port']
                    per_port[p]['objects'] += 1
                    if row['facility_kind']:
                        per_port[p]['facility_hits'] += 1
                    if row.get('facility_osm_id'):
                        per_port[p]['facility_with_osm_id'] += 1
                    if row.get('facility_edit_ts'):
                        per_port[p]['facility_dated'] += 1
                    if row.get('facility_temporal_valid') == '1':
                        per_port[p]['facility_existed_at_scene'] += 1
                    if row['on_fairway'] == '1':
                        per_port[p]['on_fairway'] += 1
                    if row['channel_angle_source']:
                        per_port[p]['channel_angle_' + row['channel_angle_source']] += 1
                    support[p][row['support_i']] += 1
    summary = out.parent / 'objects_by_port_summary.csv'
    cols = ['objects', 'facility_hits', 'facility_with_osm_id', 'facility_dated', 'facility_existed_at_scene',
            'on_fairway', 'channel_angle_osm_fairway', 'channel_angle_ais_traffic'] + ['support_%d' % i for i in range(7)]
    with summary.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['port'] + cols)
        for port in sorted(per_port):
            row = per_port[port]
            w.writerow([port] + [row[c] for c in cols[:8]] +
                       [support[port][str(i)] for i in range(7)])
    print('合并 %d 个分片 -> %s（%d 对象，%.0fs）' % (len(shards), out, total, time.time() - t0), flush=True)
    print('分港口汇总 -> %s' % summary, flush=True)


if __name__ == '__main__':
    main()

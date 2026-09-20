"""Fill facility_edit_ts / facility_temporal_valid in an existing object table from osm_way_dates.csv.

The object table can run before the OSM-API dating pass finishes; this streaming rewrite joins by
facility_osm_id afterwards, so nobody has to re-run the 300-product pass. support_i is untouched
(its six signals deliberately exclude the temporal evidence).

  python join_facility_dates.py objects/objects.csv.gz [--out objects/objects.dated.csv.gz]
"""
import argparse, csv, gzip, io, shutil, time
from pathlib import Path

KS = Path(r'E:/临时会话/knowledge_set_841')
DATES = KS / 'facilities' / 'osm_way_dates.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    out = args.out or args.table.with_suffix('.dated.csv.gz')

    dates = {}
    if DATES.is_file():
        with DATES.open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r.get('status') == 'ok' and r.get('timestamp'):
                    dates[r['osm_id']] = r['timestamp']
    print('已定年 way: %d' % len(dates), flush=True)

    t0 = time.time()
    filled = 0
    total = 0
    with gzip.open(args.table, 'rt', encoding='utf-8-sig', newline='') as src, \
         gzip.open(out, 'wt', encoding='utf-8-sig', newline='') as dst:
        reader = csv.DictReader(src)
        writer = None
        for row in reader:
            total += 1
            oid = row.get('facility_osm_id') or ''
            ts = dates.get(oid, '')
            if ts:
                scene = (row.get('start_utc') or '')[:10]
                row['facility_edit_ts'] = ts
                row['facility_temporal_valid'] = int(ts[:10] <= scene) if scene else ''
                filled += 1
            if writer is None:
                writer = csv.DictWriter(dst, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
    print('对象 %d，回填时间证据 %d (%.1f%%)，用时 %.0fs -> %s' % (
        total, filled, 100.0 * filled / max(1, total), time.time() - t0, out), flush=True)
    if out != args.table:
        print('原表未改动；确认后可自行替换', flush=True)


if __name__ == '__main__':
    main()

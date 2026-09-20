"""Recompute the AIS-class columns of an existing object table from mmsi_class_final.csv.

The object table may have been built before the 19c027 MMSI class layer landed (or before it was
wired in), so this streams a rewrite instead of re-running the 329-product build:

  ais_final_class / ais_class_level / ais_class_confidence / ais_class_source  <- the class layer
  fine_class / fine_class_source                                             <- kept in sync
  fine_class_resolved signal, support_i, support_missing                      <- recomputed

  python join_ais_class.py objects/objects_all.csv.gz
"""
import argparse, csv, gzip, sys, time
from pathlib import Path

KS = Path(r'E:/临时会话/knowledge_set_841')
CLASSES = KS / 'mmsi' / 'mmsi_class_final.csv'
SIGNALS = ['ais_unique_match', 'fine_class_resolved', 'valid_area_ok', 'coast_known',
           'facility_context', 'facility_map_available']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    out = args.out or args.table.with_name(args.table.stem + '_classed.csv.gz')

    table = {}
    with CLASSES.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('final_class'):
                table[r['mmsi']] = (r['final_class'], r.get('class_level', ''), r.get('confidence', ''),
                                    r.get('source', ''))
    print('类别层 MMSI: %d' % len(table), flush=True)

    t0 = time.time()
    n = upgraded = 0
    with gzip.open(args.table, 'rt', encoding='utf-8-sig', newline='') as src, \
         gzip.open(out, 'wt', encoding='utf-8-sig', newline='') as dst:
        reader = csv.DictReader(src)
        writer = None
        for row in reader:
            n += 1
            mmsi = row.get('matched_mmsi') or ''
            cls = table.get(mmsi)
            if cls:
                row['ais_final_class'], row['ais_class_level'], row['ais_class_confidence'], row['ais_class_source'] = cls
                row['fine_class'], row['fine_class_source'] = cls[0], cls[3]
                upgraded += 1
            elif row.get('ais_class_level'):
                row['ais_final_class'] = row['ais_class_level'] = row['ais_class_confidence'] = row['ais_class_source'] = ''
                row['fine_class'] = row['fine_class_source'] = ''
            if 'support_i' in row:
                resolved = row.get('ais_class_level', '') == 'fine' and not row.get('ais_final_class', '').endswith('_coarse')
                row['fine_class_resolved'] = int(resolved)
                row['support_i'] = sum(1 for s in SIGNALS if str(row.get(s, '0')) == '1')
                row['support_missing'] = ';'.join(s for s in SIGNALS if str(row.get(s, '0')) != '1')
            if writer is None:
                writer = csv.DictWriter(dst, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
    print('对象 %d，类表命中并回填 %d (%.1f%%)，用时 %.0fs -> %s' % (
        n, upgraded, 100.0 * upgraded / max(1, n), time.time() - t0, out), flush=True)


if __name__ == '__main__':
    main()

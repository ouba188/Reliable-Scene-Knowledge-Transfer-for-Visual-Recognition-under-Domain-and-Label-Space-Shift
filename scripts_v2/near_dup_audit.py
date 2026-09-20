"""Near-duplicate audit: same object detected twice (cross-polarisation / overlapping tiles).

The object table's exact-coordinate check found only 301 repeats, but the protocol's concern is
near duplicates (VV and VH detections of one ship land within a few metres). Uses a per-product
KD-tree and reports pairs within TOL_M, split by polarisation and tile.

  python near_dup_audit.py [objects_final.csv.gz] --out <json>
"""
import argparse, csv, gzip, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TOL_M = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table', nargs='?', type=Path,
                    default=Path(r'E:/临时会话/knowledge_set_841/objects/objects_final.csv.gz'))
    ap.add_argument('--out', type=Path,
                    default=Path(r'E:/Docms/无监督域适应/translated/0917/next_experiment_v03/e00/near_dup_report.json'))
    ap.add_argument('--tol', type=float, default=TOL_M)
    args = ap.parse_args()

    per_product = defaultdict(lambda: {'xy': [], 'pol': [], 'tile': []})
    n = 0
    with gzip.open(args.table, 'rt', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            n += 1
            d = per_product[row['product_id']]
            d['xy'].append((float(row['world_x']), float(row['world_y'])))
            d['pol'].append(row['polarization'])
            d['tile'].append(row['roi_origin'])
    print('rows %d  products %d' % (n, len(per_product)), flush=True)

    total_pairs = 0
    kinds = Counter()
    by_port_pairs = Counter()
    pair_products = 0
    for i, (pid, d) in enumerate(per_product.items()):
        xy = np.asarray(d['xy'])
        if len(xy) < 2:
            continue
        tree = cKDTree(xy)
        pairs = tree.query_pairs(args.tol, output_type='ndarray')
        if not len(pairs):
            continue
        pair_products += 1
        total_pairs += len(pairs)
        a, b = pairs[:, 0], pairs[:, 1]
        same_pol = np.array([d['pol'][i_] == d['pol'][j_] for i_, j_ in zip(a, b)])
        same_tile = np.array([d['tile'][i_] == d['tile'][j_] for i_, j_ in zip(a, b)])
        kinds['cross_pol'] += int((~same_pol).sum())
        kinds['same_pol_cross_tile'] += int((same_pol & ~same_tile).sum())
        kinds['same_pol_same_tile'] += int((same_pol & same_tile).sum())
        if (i + 1) % 50 == 0:
            print('  %d/%d products, pairs %d' % (i + 1, len(per_product), total_pairs), flush=True)

    out = dict(
        table=str(args.table), rows=n, products=len(per_product), tolerance_m=args.tol,
        pairs_within_tolerance=total_pairs,
        products_with_duplicates=pair_products,
        duplicate_share_of_rows=round(2 * total_pairs / max(1, n), 4),
        breakdown=dict(kinds),
        interpretation=[
            'cross_pol pairs are the VV/VH double detections; dedupe before computing density, '
            'nearest-neighbour distance and heading statistics',
            'same_pol_same_tile pairs are detector double-fires inside one tile',
            'same_pol_cross_tile pairs come from overlapping windows',
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print('->', args.out)


if __name__ == '__main__':
    main()

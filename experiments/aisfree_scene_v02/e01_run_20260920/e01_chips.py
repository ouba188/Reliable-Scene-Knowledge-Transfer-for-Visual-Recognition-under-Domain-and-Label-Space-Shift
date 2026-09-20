"""E01 chip extraction (server side).

Spec (config.json / docs/CANDIDATE_BANK.md):
  - one dedup object = one instance row; VV+VH are the 2 channels of ONE chip
  - fixed native-resolution axis-aligned 128x128 window centred on the dedup centroid
  - NO resize / no OBB-aligned normalisation; both pols same geospatial window
  - uint8 display raster; validity = non-zero; record valid_fraction and clipping
  - output: per-product .npz shards (uint8 [N,2,128,128]) + index csv

usage:
  python e01_chips.py --index            # build raster index only (inventory)
  python e01_chips.py --smoke PORT       # one port, verbose
  python e01_chips.py --all              # full run, resumable
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
OUT = ROOT / 'e01_chips'
HALF = 64
SIZE = 128
PROD_RE = re.compile(r'(S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4})_([VH]{2})')

PREFER = ['safe841_batch/parallel', 'safe841_batch/annotation_output', 'fdrive_import_20260920',
          'asf_download_v2_allports', 'asf_download_v2_tmp', 'sar_download_v1', 'safe841_batch_s3']


def build_index(products: set[str] | None = None) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], list[str]] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(ROOT):
        if '/knowledge_841' in dirpath or '/phase0' in dirpath or '/e01_chips' in dirpath:
            continue
        for f in files:
            if not f.endswith('.tif'):
                continue
            m = PROD_RE.search(f)
            if not m:
                continue
            prod, pol = m.group(1), m.group(2)
            if products and prod not in products:
                continue
            found[(prod, pol)].append(os.path.join(dirpath, f))

    def rank(p: str) -> int:
        for i, pref in enumerate(PREFER):
            if pref in p:
                return i
        return len(PREFER)

    idx = {k: sorted(v, key=rank)[0] for k, v in found.items()}
    return idx


def load_objects(products: set[str], ports: set[str] | None, limit: int | None):
    """P0 offshore, dedup representatives only."""
    rows = []
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            if r['product_id'] not in products:
                continue
            if r.get('deployable_offshore') != '1':
                continue
            if ports and r['port'] not in ports:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def crop_pair(paths: dict[str, str], x: float, y: float):
    """Return chips[2,128,128] uint8 (0 where outside) + valid fraction + clipped flag."""
    chips = np.zeros((2, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros((SIZE, SIZE), dtype=bool)
    clipped = False
    for ch, pol in enumerate(('VV', 'VH')):
        p = paths.get(pol)
        if p is None:
            return None, 0.0, True
        with rasterio.open(p) as ds:
            col, row = ds.index(x, y)  # uses the dataset's own transform
            win = Window(col - HALF, row - HALF, SIZE, SIZE)
            try:
                need = win.intersection(Window(0, 0, ds.width, ds.height))
            except Exception:
                clipped = True
                continue  # this pol contributes nothing; other pol may still be inside
            if need.width < SIZE or need.height < SIZE:
                clipped = True
            data = ds.read(1, window=need, boundless=True, fill_value=0, out_shape=(need.height, need.width))
            offy = int(need.row_off - win.row_off)
            offx = int(need.col_off - win.col_off)
            h, w = data.shape
            chips[ch, offy:offy + h, offx:offx + w] = data
    nonzero = (chips[0] > 0) | (chips[1] > 0)
    valid |= nonzero
    vf = float(valid.mean())
    return chips, vf, clipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--smoke', metavar='PORT')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--products', help='csv of product ids (default: manifests/products_329.csv)')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--shard', help='i/n -> process only products where index%%n==i')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()

    prod_csv = Path(args.products) if args.products else KS / 'e01_inputs' / 'products_329.csv'
    products = {r['product_id'] if 'product_id' in r else r.get('product') or list(r.values())[0]
                for r in csv.DictReader(prod_csv.open(encoding='utf-8'))}
    products = {p.strip() for p in products if p and p.strip()}
    print('冻结产品数:', len(products), flush=True)

    idx = build_index(products)
    print('栅格索引: %d 个 (产品,极化)' % len(idx), flush=True)
    miss = [(p, pol) for p in list(products)[:2] for pol in ('VV', 'VH') if (p, pol) not in idx]
    if args.index:
        for k in list(idx)[:5]:
            print('  样例', k, '->', idx[k])
        return

    ports = {args.smoke} if args.smoke else None
    objs = load_objects(products, ports, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        prods = sorted({r['product_id'] for r in objs})
        keep = {p for k, p in enumerate(sorted(products)) if k % n == i}
        objs = [r for r in objs if r['product_id'] in keep]
        print('分片 %s: 产品 %d 个, 对象 %d' % (args.shard, len(keep), len(objs)), flush=True)
    print('对象数:', len(objs), flush=True)
    by_prod = defaultdict(list)
    for r in objs:
        by_prod[r['product_id']].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    stats = []
    for i, (prod, rows) in enumerate(sorted(by_prod.items()), 1):
        outp = args.out / (prod + '.npz')
        if outp.exists():
            done += 1
            continue
        paths = {pol: idx[(prod, pol)] for pol in ('VV', 'VH') if (prod, pol) in idx}
        if len(paths) < 2:
            print('  缺极化:', prod, sorted(paths), flush=True)
            continue
        chips = np.zeros((len(rows), 2, SIZE, SIZE), dtype=np.uint8)
        vfs = np.zeros(len(rows), dtype=np.float32)
        clippeds = np.zeros(len(rows), dtype=np.uint8)
        keep = []
        for j, r in enumerate(rows):
            c, vf, clip = crop_pair(paths, float(r['world_x']), float(r['world_y']))
            if c is None:
                continue
            chips[j] = c
            vfs[j] = vf
            clippeds[j] = 1 if clip else 0
            keep.append(j)
        sample_ids = np.array([rows[j]['object_id'] for j in keep])
        np.savez_compressed(outp, chips=chips[keep], valid_fraction=vfs[keep],
                            clipped=clippeds[keep], sample_id=sample_ids)
        stats.append((prod, len(keep), float((vfs[keep] >= 0.95).mean()), float(vfs[keep].mean()), int(clippeds[keep].sum())))
        done += 1
        if args.smoke or i % 20 == 0:
            print('  [%d/%d] %s  对象 %d  有效>=.95 %.1f%% 均值 %.3f 越界 %d  (%.0fs)' % (
                i, len(by_prod), prod[-17:], len(keep), 100 * stats[-1][2], stats[-1][3], stats[-1][4], time.time() - t0), flush=True)

    with (args.out / 'chip_index.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['product_id', 'objects', 'frac_valid_ge_095', 'mean_valid_fraction', 'clipped'])
        w.writerows(stats)
    print('完成 %d 个产品分片 -> %s  (%.0f 分钟)' % (done, args.out, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()

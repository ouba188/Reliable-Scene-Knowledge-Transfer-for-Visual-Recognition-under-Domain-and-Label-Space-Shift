"""E01 chip extraction (server side) — revision 1 (2026-09-21).

Fixes after ChatGPT's execution review:
  * rasterio's ds.index(x, y) returns (row, col) — rev0 had them swapped, so all rev0 chips are
    at transposed positions and are kept only as a discarded pilot.
  * validity is now explicit: geometric inside-raster fraction per polarisation plus the joint
    fraction; the nonzero fraction is recorded separately (raw 0 is NOT treated as invalid).
  * files are written to a temp name and atomically renamed; a per-shard DONE marker plus a
    global _COMPLETE marker make readiness detectable without reading half-written npz.

usage:
  python e01_chips.py --index
  python e01_chips.py --smoke "Jebel Ali"
  python e01_chips.py --all [--shard i/n]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
OUT = ROOT / 'e01_chips_rev1'
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

    return {k: sorted(v, key=rank)[0] for k, v in found.items()}


def load_objects(products: set[str], ports: set[str] | None, limit: int | None):
    rows = []
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['product_id'] not in products or r.get('deployable_offshore') != '1':
                continue
            if ports and r['port'] not in ports:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def crop_pair(paths: dict[str, str], x: float, y: float):
    """Return chips[2,128,128] uint8, per-pol geometric validity, joint validity, nonzero fraction.

    Validity = fraction of the 128x128 window that falls inside the raster extent (row/col order
    per rasterio: ds.index returns (row, col)). Raw zero pixels stay valid; they are reported
    separately as a nonzero fraction so nothing implicit is assumed about radiometry.
    """
    chips = np.zeros((2, SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros((2, SIZE, SIZE), dtype=bool)
    clipped = 0
    for ch, pol in enumerate(('VV', 'VH')):
        p = paths.get(pol)
        if p is None:
            return None, np.zeros(2), 0.0, 0.0, 1
        with rasterio.open(p) as ds:
            row, col = ds.index(x, y)          # rasterio returns (row, col)
            win = Window(col - HALF, row - HALF, SIZE, SIZE)
            try:
                need = win.intersection(Window(0, 0, ds.width, ds.height))
            except Exception:
                clipped = 1
                continue
            if need.width < SIZE or need.height < SIZE:
                clipped = 1
            data = ds.read(1, window=need, boundless=True, fill_value=0,
                           out_shape=(int(need.height), int(need.width)))
            oy = int(need.row_off - win.row_off)
            ox = int(need.col_off - win.col_off)
            h, w = data.shape
            chips[ch, oy:oy + h, ox:ox + w] = data
            valid[ch, oy:oy + h, ox:ox + w] = True
    per_pol = valid.mean(axis=(1, 2))
    joint = float(valid.all(axis=0).mean())
    nz = float(((chips[0] > 0) | (chips[1] > 0)).mean())
    return chips, per_pol, joint, nz, clipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--smoke', metavar='PORT')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--products', help='csv of product ids (default: e01_inputs/products_329.csv)')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--shard', help='i/n')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()

    prod_csv = Path(args.products) if args.products else KS / 'e01_inputs' / 'products_329.csv'
    products = set()
    with prod_csv.open(encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            v = r.get('product_id') or list(r.values())[0]
            if v and v.strip():
                products.add(v.strip())
    print('冻结产品数:', len(products), flush=True)

    idx = build_index(products)
    print('栅格索引: %d 个 (产品,极化)' % len(idx), flush=True)
    if args.index:
        for k in list(idx)[:5]:
            print('  样例', k, '->', idx[k])
        return

    ports = {args.smoke} if args.smoke else None
    objs = load_objects(products, ports, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        keep = {p for k, p in enumerate(sorted(products)) if k % n == i}
        objs = [r for r in objs if r['product_id'] in keep]
    print('对象数:', len(objs), flush=True)
    by_prod = defaultdict(list)
    for r in objs:
        by_prod[r['product_id']].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / '_COMPLETE').unlink(missing_ok=True)
    t0 = time.time()
    done = 0
    stats = []
    for i, (prod, rows) in enumerate(sorted(by_prod.items()), 1):
        outp = args.out / (prod + '.npz')
        donep = args.out / (prod + '.done')
        if outp.exists() and donep.exists():
            done += 1
            continue
        paths = {pol: idx[(prod, pol)] for pol in ('VV', 'VH') if (prod, pol) in idx}
        if len(paths) < 2:
            print('  缺极化:', prod, sorted(paths), flush=True)
            continue
        chips = np.zeros((len(rows), 2, SIZE, SIZE), dtype=np.uint8)
        per_pol = np.zeros((len(rows), 2), dtype=np.float32)
        joint = np.zeros(len(rows), dtype=np.float32)
        nz = np.zeros(len(rows), dtype=np.float32)
        clip = np.zeros(len(rows), dtype=np.uint8)
        keep = []
        for j, r in enumerate(rows):
            c, pp, jv, zf, cl = crop_pair(paths, float(r['world_x']), float(r['world_y']))
            if c is None:
                continue
            chips[j] = c
            per_pol[j] = pp
            joint[j] = jv
            nz[j] = zf
            clip[j] = cl
            keep.append(j)
        tmp = args.out / (prod + '.npz.tmp')
        # np.savez_compressed appends .npz to a path without that suffix, so hand it a file object
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, chips=chips[keep], valid_joint=joint[keep], valid_vv=per_pol[keep, 0],
                                valid_vh=per_pol[keep, 1], nonzero_fraction=nz[keep], clipped=clip[keep],
                                sample_id=np.array([rows[j]['object_id'] for j in keep], dtype='<U128'))
        tmp.rename(outp)
        donep.write_text('ok\n')
        stats.append((prod, len(keep), float((joint[keep] >= 0.95).mean()), float(joint[keep].mean()),
                      int(clip[keep].sum()), float(per_pol[keep, 0].mean()), float(per_pol[keep, 1].mean())))
        done += 1
        if args.smoke or i % 20 == 0:
            s = stats[-1]
            print('  [%d/%d] %s 对象 %d 联合有效>=.95 %.1f%% 均值 %.3f 越界 %d (VV均 %.3f VH均 %.3f) (%.0fs)' % (
                i, len(by_prod), prod[-17:], s[1], 100 * s[2], s[3], s[4], s[5], s[6], time.time() - t0), flush=True)

    with (args.out / 'chip_index.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['product_id', 'objects', 'frac_valid_ge_095', 'mean_valid', 'clipped',
                    'mean_valid_vv', 'mean_valid_vh'])
        w.writerows(stats)
    (args.out / '_COMPLETE').write_text('shards=%d\nfinished=%s\n' % (done, time.strftime('%Y-%m-%d %H:%M:%S')))
    print('完成 %d 个产品分片 -> %s  (%.0f 分钟)' % (done, args.out, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()

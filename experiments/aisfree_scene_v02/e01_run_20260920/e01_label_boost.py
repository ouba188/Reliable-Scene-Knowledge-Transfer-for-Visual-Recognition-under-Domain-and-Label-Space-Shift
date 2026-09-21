"""E01 auxiliary label boost: spatiotemporal AIS matching for P0 objects of target products.

Produces a SEPARATE label tier (never touches the frozen main labels):
  T1 strict : exactly one MMSI within 150 m and +-180 s of the scene epoch
  T2 relaxed: exactly one MMSI within 500 m and +-300 s
Class comes from the MMSI class layer (fine preferred, coarse recorded separately).

usage: python e01_label_boost.py --folds "Shanghai,Rotterdam,Port Klang,Fujairah,Jebel Ali,Port Said"
"""
from __future__ import annotations

import argparse
import csv
import os
import gzip
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
AIS_ROOT = ROOT / 'safe841_batch' / 'annotation_bundle_s3' / 'assets' / 'products'
OUT = ROOT / 'e01_label_boost'
PROD_RE = re.compile(r'(S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4})_([VH]{2})')
PREFER = ['safe841_batch/parallel', 'safe841_batch/annotation_output', 'fdrive_import_20260920',
          'asf_download_v2_allports', 'asf_download_v2_tmp', 'sar_download_v1', 'safe841_batch_s3']
T1 = (150.0, 180)
T2 = (500.0, 300)


def raster_index(products: set[str]) -> dict[str, str]:
    found: dict[tuple[str, str], list[str]] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(ROOT):
        if '/knowledge_841' in dirpath or '/phase0' in dirpath or '/e01_' in dirpath:
            continue
        for f in files:
            if not f.endswith('.tif'):
                continue
            m = PROD_RE.search(f)
            if m and m.group(1) in products:
                found[(m.group(1), m.group(2))].append(dirpath + '/' + f)

    def rank(p: str) -> int:
        for i, pref in enumerate(PREFER):
            if pref in p:
                return i
        return len(PREFER)
    idx = {}
    for (prod, pol), paths in found.items():
        if pol == 'VV':
            idx[prod] = sorted(paths, key=rank)[0]
    return idx


def load_class_layer() -> dict[str, tuple[str, str]]:
    out = {}
    with (KS / 'mmsi' / 'mmsi_class_final.csv').open(encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            out[str(r['mmsi']).strip()] = (r.get('final_class') or '', r.get('class_level') or '')
    return out


def load_objects(products: set[str]):
    per = defaultdict(list)
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1' and r['product_id'] in products:
                per[r['product_id']].append(r)
    return per


def read_ais(prod: str):
    rows = []
    for f in sorted((AIS_ROOT / prod / 'ais').glob('*.txt')):
        with f.open(errors='ignore') as fh:
            for line in fh:
                p = line.rstrip('\n').split(',')
                if len(p) < 8:
                    continue
                try:
                    rows.append((float(p[0]), p[1].strip(), float(p[6]), float(p[7])))
                except ValueError:
                    continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', required=True)
    ap.add_argument('--manifest', type=Path, default=KS / 'e01_first_batch' / 'split_manifest.json')
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    folds = [f.strip() for f in a.folds.split(',')]
    prods_by_fold = {}
    all_prods = set()
    for t in folds:
        f = man['folds'][t]
        ps = sorted(set(f['target_adapt_products']) | set(f['target_eval_products']))
        prods_by_fold[t] = ps
        all_prods |= set(ps)
    print('折数 %d, 目标产品合计 %d' % (len(folds), len(all_prods)), flush=True)

    idx = raster_index(all_prods)
    print('栅格索引 %d / %d' % (len(idx), len(all_prods)), flush=True)
    cls = load_class_layer()
    print('MMSI 类层 %d 条' % len(cls), flush=True)
    objs = load_objects(all_prods)
    print('目标产品 P0 对象合计 %d' % sum(len(v) for v in objs.values()), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    t0 = time.time()
    for k, prod in enumerate(sorted(all_prods), 1):
        outp = OUT / (prod + '.csv.gz')
        if outp.exists():
            continue
        rows = objs.get(prod, [])
        if not rows:
            continue
        ais = read_ais(prod)
        if not ais:
            continue
        ep = np.array([r[0] for r in ais])
        mm = np.array([r[1] for r in ais])
        lon = np.array([r[2] for r in ais])
        lat = np.array([r[3] for r in ais])
        try:
            crs = rasterio.open(idx[prod]).crs
        except Exception:
            continue
        tr = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        ax, ay = tr.transform(lon, lat)
        ax = np.asarray(ax); ay = np.asarray(ay)
        scene_epoch = None
        if rows[0].get('start_utc'):
            import datetime as dt
            scene_epoch = dt.datetime.strptime(rows[0]['start_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc).timestamp()
        if scene_epoch is None:
            continue
        keep = np.abs(ep - scene_epoch) <= 300
        if keep.sum() == 0:
            continue
        tree = cKDTree(np.c_[ax[keep], ay[keep]])
        kep_e, kep_m = ep[keep], mm[keep]
        out_rows = []
        for r in rows:
            x, y = float(r['world_x']), float(r['world_y'])
            d, i = tree.query([x, y], k=8, distance_upper_bound=500.0)
            cand = [(dd, ii) for dd, ii in zip(np.atleast_1d(d), np.atleast_1d(i))
                    if np.isfinite(dd) and ii < len(kep_m)]
            t1 = t2 = ''
            mmsi = ''
            dist = dt_s = ''
            ncand = len({kep_m[ii] for _dd, ii in cand})
            for lim_d, lim_t in (T1, T2):
                ok = [(dd, ii) for dd, ii in cand if dd <= lim_d and abs(kep_e[ii] - scene_epoch) <= lim_t]
                uniq = {kep_m[ii] for _dd, ii in ok}
                if len(uniq) == 1:
                    m = uniq.pop()
                    c, lvl = cls.get(m, ('', ''))
                    lab = c if (c and (lvl == 'fine' or c.endswith('_coarse'))) else ''
                    if lim_d == T1[0]:
                        t1 = lab; mmsi = m if not t2 else mmsi
                        if lab:
                            dist = ok[0][0]; dt_s = abs(float(kep_e[ok[0][1]]) - scene_epoch)
                    else:
                        t2 = lab
                    if t1:
                        break
            out_rows.append((r['object_id'], t1, t2, mmsi if t1 else '', dist, dt_s, ncand))
        with gzip.open(outp, 'wt', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['object_id', 'boost_t1_class', 'boost_t2_class', 'boost_mmsi', 'boost_distance_m',
                        'boost_dt_s', 'boost_n_candidate_mmsi'])
            w.writerows(out_rows)
        n1 = sum(1 for r in out_rows if r[1])
        n2 = sum(1 for r in out_rows if r[1] or r[2])
        summary.append((prod, len(out_rows), n1, n2, len(ais)))
        if k % 20 == 0 or k == len(all_prods):
            print('  [%d/%d] %s t1=%d t2=%d (%.0fs)' % (k, len(all_prods), prod[-9:], n1, n2, time.time() - t0), flush=True)

    with (OUT / 'boost_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['product_id', 'objects', 't1_labelled', 't2_labelled', 'ais_rows'])
        w.writerows(summary)

    # per-fold yield over eval / adapt sets
    fold_rep = {}
    for t in folds:
        f = man['folds'][t]
        for role in ('target_eval', 'target_adapt'):
            prods = set(f[role + '_products'])
            tot = set()
            for p in prods:
                rp = OUT / (p + '.csv.gz')
                if not rp.exists():
                    continue
                with gzip.open(rp, 'rt', encoding='utf-8') as fh:
                    rd = csv.DictReader(fh)
                    for r in rd:
                        tot.add((r['object_id'], r['boost_t1_class'], r['boost_t2_class']))
            n1 = sum(1 for _i, a1, _a2 in tot if a1)
            n2 = sum(1 for _i, a1, a2 in tot if a1 or a2)
            fold_rep.setdefault(t, {})[role] = {'instances': len(tot), 't1': n1, 't2': n2}
    (OUT / 'boost_fold_yield.json').write_text(json.dumps(fold_rep, indent=1, ensure_ascii=False))
    print(json.dumps(fold_rep, ensure_ascii=False, indent=1)[:1500])


if __name__ == '__main__':
    main()

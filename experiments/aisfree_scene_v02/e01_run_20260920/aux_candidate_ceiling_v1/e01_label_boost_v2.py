"""Boost label tier v2 — complete-radius candidate enumeration (correction of v1).

v1 bug: cKDTree.query(k=8) returns at most 8 records, so a neighbouring MMSI could hide inside the
radius and the object was wrongly called "single candidate". v2 uses query_ball_point(r=500) for the
COMPLETE candidate set, then applies the reviewer's tier reference (match_tiers_reference.py).

Per tier (T1 150 m / 180 s, T2 500 m / 300 s) it stores independent evidence:
  matched_mmsi, distance_m, abs_dt_s, n_candidate_records, n_candidate_mmsi, status
plus the resolved class (fine preferred from the MMSI class layer) and a (product_id, MMSI) reuse flag.

usage: python e01_label_boost_v2.py --folds "Shanghai,Rotterdam,Port Klang,Fujairah,Jebel Ali,Port Said"
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

sys_path = Path('/root/autodl-tmp/knowledge_841_20260920/e01_inputs')
import sys
sys.path.insert(0, str(sys_path))
from match_tiers_reference import match_tiers_from_complete_radius  # noqa: E402

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
AIS_ROOT = ROOT / 'safe841_batch' / 'annotation_bundle_s3' / 'assets' / 'products'
OUT = ROOT / 'e01_label_boost_v2'
RADIUS = 500.0
PROD_RE = __import__('re').compile(r'(S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4})_([VH]{2})')
PREFER = ['safe841_batch/parallel', 'safe841_batch/annotation_output', 'fdrive_import_20260920',
          'asf_download_v2_allports', 'asf_download_v2_tmp', 'sar_download_v1', 'safe841_batch_s3']


def raster_index(products):
    found = defaultdict(list)
    for dirpath, _dirs, files in os.walk(ROOT):
        if '/knowledge_841' in dirpath or '/phase0' in dirpath or '/e01_' in dirpath:
            continue
        for f in files:
            if f.endswith('.tif'):
                m = PROD_RE.search(f)
                if m and m.group(1) in products:
                    found[(m.group(1), m.group(2))].append(dirpath + '/' + f)

    def rank(p):
        for i, pref in enumerate(PREFER):
            if pref in p:
                return i
        return len(PREFER)
    return {k[0]: sorted(v, key=rank)[0] for k, v in found.items() if k[1] == 'VV'}


def load_class_layer():
    out = {}
    with (KS / 'mmsi' / 'mmsi_class_final.csv').open(encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            out[str(r['mmsi']).strip()] = (r.get('final_class') or '', r.get('class_level') or '')
    return out


def read_ais(prod):
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
    print('折 %d, 目标产品 %d' % (len(folds), len(all_prods)), flush=True)

    idx = raster_index(all_prods)
    cls = load_class_layer()
    print('栅格 %d, MMSI 类层 %d' % (len(idx), len(cls)), flush=True)

    objs = defaultdict(list)
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1' and r['product_id'] in all_prods:
                objs[r['product_id']].append(r)
    print('目标产品 P0 对象 %d' % sum(len(v) for v in objs.values()), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stats = []
    mmsi_reuse = Counter()
    t0 = time.time()
    for k, prod in enumerate(sorted(all_prods), 1):
        outp = OUT / (prod + '.csv.gz')
        if outp.exists():
            continue
        rows = objs.get(prod, [])
        if not rows or prod not in idx:
            continue
        ais = read_ais(prod)
        if not ais:
            continue
        ep = np.array([r[0] for r in ais]); mm = np.array([r[1] for r in ais])
        lon = np.array([r[2] for r in ais]); lat = np.array([r[3] for r in ais])
        try:
            crs = rasterio.open(idx[prod]).crs
        except Exception:
            continue
        ax, ay = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
        import datetime as dt
        scene_epoch = dt.datetime.strptime(rows[0]['start_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(
            tzinfo=dt.timezone.utc).timestamp()
        window = np.abs(ep - scene_epoch) <= 300.0          # same convention as v1: +-300 s
        if not window.any():
            continue
        XY = np.c_[np.asarray(ax)[window], np.asarray(ay)[window]]
        KE, KM = ep[window], mm[window]
        tree = cKDTree(XY)
        out_rows = []
        for r in rows:
            xy = np.array([float(r['world_x']), float(r['world_y'])])
            cand = np.asarray(tree.query_ball_point(xy, r=RADIUS, eps=0.0), dtype=np.int64)
            if len(cand):
                d = np.linalg.norm(XY[cand] - xy, axis=1)
                tiers = match_tiers_from_complete_radius(d, KE[cand] - scene_epoch, KM[cand])
            else:
                tiers = {'T1': {'matched_mmsi': None, 'distance_m': None, 'abs_dt_s': None,
                                'n_candidate_records': 0, 'n_candidate_mmsi': 0, 'status': 'no_candidate'},
                         'T2': {'matched_mmsi': None, 'distance_m': None, 'abs_dt_s': None,
                                'n_candidate_records': 0, 'n_candidate_mmsi': 0, 'status': 'no_candidate'}}
            def lab(t):
                m = tiers[t]['matched_mmsi']
                if not m:
                    return ''
                c, lvl = cls.get(m, ('', ''))
                return c if (c and (lvl == 'fine' or c.endswith('_coarse'))) else ''
            t1, t2 = tiers['T1'], tiers['T2']
            if t1['matched_mmsi']:
                mmsi_reuse[(prod, t1['matched_mmsi'])] += 1
            out_rows.append((r['object_id'], lab('T1'), lab('T2'),
                             t1['matched_mmsi'] or '', t1['distance_m'] or '', t1['abs_dt_s'] or '',
                             t1['n_candidate_records'], t1['n_candidate_mmsi'], t1['status'],
                             t2['matched_mmsi'] or '', t2['distance_m'] or '', t2['abs_dt_s'] or '',
                             t2['n_candidate_records'], t2['n_candidate_mmsi'], t2['status']))
        tmp = OUT / (prod + '.csv.gz.tmp')
        with gzip.open(tmp, 'wt', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['object_id', 'v2_t1_class', 'v2_t2_class',
                        't1_mmsi', 't1_distance_m', 't1_abs_dt_s', 't1_n_records', 't1_n_mmsi', 't1_status',
                        't2_mmsi', 't2_distance_m', 't2_abs_dt_s', 't2_n_records', 't2_n_mmsi', 't2_status'])
            w.writerows(out_rows)
        tmp.rename(outp)
        n1 = sum(1 for r in out_rows if r[1]); n2 = sum(1 for r in out_rows if r[1] or r[2])
        only2 = n2 - n1
        stats.append((prod, len(out_rows), n1, only2, n2,
                      sum(1 for r in out_rows if r[8] == 'ambiguous'), sum(1 for r in out_rows if r[14] == 'ambiguous')))
        if k % 20 == 0 or k == len(all_prods):
            print('  [%d/%d] 产品 t1=%d t2only=%d (%.0fs)' % (k, len(all_prods), n1, only2, time.time() - t0), flush=True)

    with (OUT / 'boost_v2_summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['product_id', 'objects', 't1', 't2_only', 't1plus_t2_union', 'ambiguous_t1', 'ambiguous_t2'])
        w.writerows(stats)
    reuse = {k: v for k, v in mmsi_reuse.items() if v > 1}
    (OUT / 'mmsi_reuse_check.json').write_text(json.dumps(
        {'pairs_with_multiple_objects': len(reuse), 'max_objects_per_pair': max(reuse.values()) if reuse else 0,
         'top': [{'product': p[-9:], 'mmsi': m, 'objects': c} for (p, m), c in
                 sorted(reuse.items(), key=lambda kv: -kv[1])[:10]]}, ensure_ascii=False, indent=1))
    print('完成: %d 产品; (product,MMSI) 复用对 %d' % (len(stats), len(reuse)), flush=True)
    print('T1 合计 %d, T2-only 合计 %d' % (sum(s[2] for s in stats), sum(s[3] for s in stats)), flush=True)


if __name__ == '__main__':
    main()

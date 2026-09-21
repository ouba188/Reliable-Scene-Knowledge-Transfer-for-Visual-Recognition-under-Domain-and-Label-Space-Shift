"""E06 step 1: whitelist audit of ADDED source supervision per fold (no training, CPU only).

Rules enforced (reviewer):
  * only products already in that fold's source_fit role may contribute;
  * meta / calibration-role labels never migrate into fit; no cross-fold merging;
  * no target or pseudo-target labels, no AIS identity from adapt/eval, no oracle results;
  * the global boost table is NOT joined wholesale - only rows whose product is in the fold's fit list.

Candidates for added supervision (all restricted to the fit products, deployable-offshore, in-vocabulary):
  A1 = boost_v2 tier T1 (strict, 150 m / 180 s)
  A2 = boost_v2 tier T1 ∪ T2-only
  A3 = A2 restricted to rows whose (product, MMSI) pair did not already resolve in the frozen table

Outputs: source_added_manifest.csv, source_support_delta.csv, role_exclusion_check.json,
         paired_training_manifest.json, e06_audit_summary.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
BOOST = ROOT / 'e01_label_boost_v2'
OUT = ROOT / 'e06_source_supervision'
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'Fujairah', 'JebelAli', 'PortSaid']
META = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortKlang': 'Port Klang',
        'Fujairah': 'Fujairah', 'JebelAli': 'Jebel Ali', 'PortSaid': 'Port Said'}


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    manifest = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']
    OUT.mkdir(parents=True, exist_ok=True)

    objects = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                objects[r['object_id']] = r

    added_rows, delta_rows, exclusion, paired = [], [], {}, {}
    for tag in FOLDS:
        f = manifest[META[tag]]
        vocab = [str(x) for x in json.loads(
            (ROOT / 'e01_runs' / (META[tag] + '_rev1') / 'heads' / 'class_vocab.json').read_text())]
        v = set(vocab)
        fit_prods = set(f.get('source_fit_products', []))
        forbidden = (set(f.get('source_meta_query_products', [])) | set(f.get('source_calibration_products', []))
                     | set(f.get('source_calibration_adapt_products', []))
                     | set(f.get('source_calibration_shadow_eval_products', []))
                     | set(f.get('target_adapt_products', [])) | set(f.get('target_eval_products', [])))
        frozen = [o for o, r in objects.items() if r['product_id'] in fit_prods and (r.get('ais_final_class') or '').strip() in v]
        frozen_sup = Counter(objects[o]['ais_final_class'].strip() for o in frozen)

        tiers = {'A1': [], 'A2': [], 'A3': []}
        for p in sorted(fit_prods):
            fz = BOOST / (p + '.csv.gz')
            if not fz.exists():
                continue
            with gzip.open(fz, 'rt', encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    c1, c2 = r['v2_t1_class'], r['v2_t2_class']
                    for key, cls in (('A1', c1), ('A2', c1 or c2)):
                        if cls in v:
                            tiers[key].append({'fold': tag, 'product_id': p, 'object_id': r['object_id'],
                                               'added_class': cls, 'tier': key,
                                               't1_status': r['t1_status'], 't2_status': r['t2_status'],
                                               't1_mmsi': r['t1_mmsi'], 't2_mmsi': r['t2_mmsi'],
                                               't1_distance_m': r['t1_distance_m'], 't1_abs_dt_s': r['t1_abs_dt_s'],
                                               'already_in_frozen': '1' if (objects.get(r['object_id'], {}).get('ais_final_class') or '').strip() in v else '0'})
        for key in ('A1', 'A2'):
            got = {x['object_id'] for x in tiers[key]}
            tiers['A3'] = [x for x in tiers[key] if x['already_in_frozen'] == '0'] if key == 'A2' else tiers['A3']
            _ = got
        added_rows.extend(tiers['A2'])
        for key in ('A1', 'A2', 'A3'):
            rows = tiers[key]
            cls = Counter(x['added_class'] for x in rows)
            delta_rows.append({'fold': tag, 'candidate': key,
                               'frozen_support_total': sum(frozen_sup.values()),
                               'added_total': len(rows),
                               'added_new_novel': sum(1 for x in rows if x['already_in_frozen'] == '0'),
                               'frozen_classes': len(frozen_sup), 'added_classes': len(cls),
                               'per_class_added': json.dumps(dict(cls.most_common()), ensure_ascii=False)})
        # role exclusion: any added row whose product is NOT in the fit list would be a violation
        bad = [x for x in tiers['A2'] if x['product_id'] not in fit_prods]
        exclusion[tag] = {'fit_products': len(fit_prods), 'forbidden_role_products': len(forbidden),
                          'products_shared_between_fit_and_forbidden': len(fit_prods & forbidden),
                          'added_rows_from_non_fit_products': len(bad),
                          'violation': bool(bad) or bool(fit_prods & forbidden)}
        paired[tag] = {'fold': tag, 'encoder_frozen': META[tag] + '_rev1',
                       'M0_labels': 'frozen object-table labels on fit products (current source-fit)',
                       'M1_labels': 'M0 + ' + str(len(tiers['A2'])) + ' whitelisted boost_v2 rows on the SAME products',
                       'M2_labels': 'same as M1 but encoder retrained (GPU) and PCA refit',
                       'vocab': len(vocab), 'fit_products': len(fit_prods),
                       'distinct_added_products': len({x['product_id'] for x in tiers['A2']})}

    with (OUT / 'source_added_manifest.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(added_rows[0].keys())); w.writeheader(); w.writerows(added_rows)
    with (OUT / 'source_support_delta.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(delta_rows[0].keys())); w.writeheader(); w.writerows(delta_rows)
    (OUT / 'role_exclusion_check.json').write_text(json.dumps(exclusion, ensure_ascii=False, indent=1))
    (OUT / 'paired_training_manifest.json').write_text(json.dumps(paired, ensure_ascii=False, indent=1))
    print('%-10s %-4s %8s %8s %8s %8s' % ('fold', '候选', '冻结', '新增', '其中新标签', '类别数'))
    for d in delta_rows:
        print('%-10s %-4s %8d %8d %8d %8d' % (d['fold'], d['candidate'], d['frozen_support_total'],
                                              d['added_total'], d['added_new_novel'], d['added_classes']))
    print('\n角色排他检查：', json.dumps({k: v['violation'] for k, v in exclusion.items()}, ensure_ascii=False))
    tot = sum(d['added_new_novel'] for d in delta_rows if d['candidate'] == 'A3')
    print('A3（真正新增且不在冻结表内）总计 %d 条' % tot)


if __name__ == '__main__':
    main()

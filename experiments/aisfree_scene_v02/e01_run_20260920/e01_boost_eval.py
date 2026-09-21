"""E01 auxiliary evaluation with boost labels (single column, never used for training/model selection).

For each fold re-evaluates B0 / B1 / Full on the eval instances that carry a boost t1 (strict
150 m / 180 s, single-MMSI) or t2 (500 m / 300 s) label, using the stored eval probabilities.
Also reports the agreement between boost and frozen labels as an integrity line.

usage: python e01_boost_eval.py --fold Shanghai --run /root/autodl-tmp/e01_runs/Shanghai_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
BOOST = ROOT / 'e01_label_boost'
OUT = BOOST / 'eval'
MAPPING_VERSION = 'boost_t1_t2_v1'


def balanced_accuracy(y, pred):
    vals = [float((pred[y == c] == c).mean()) for c in sorted(set(y)) if (y == c).sum()]
    return float(np.mean(vals)) if vals else None


def load_frozen():
    out = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                out[r['object_id']] = (r.get('ais_final_class') or '', r.get('ais_class_level') or '')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()

    vocab = json.loads((a.run / 'heads' / 'class_vocab.json').read_text())
    ci = {c: k for k, c in enumerate(vocab)}
    probs = np.load(a.run / 'eval_probabilities.npy')
    eidx = np.load(a.run / 'eval_index.npy')
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    b0 = json.loads((a.run / 'meta_head_selection.json').read_text())['B0']
    ret = json.loads((a.run / 'adapt_filter' / 'retention.json').read_text())['retained']
    frozen = load_frozen()

    # boost labels per eval instance (t1 strict, t2 relaxed)
    per_prod_cache = {}
    y_t1 = []
    y_t2 = []
    for i in eidx:
        oid = ids[i]
        prod = oid.split('|')[0] if '|' in oid else ''
        if prod not in per_prod_cache:
            m = {}
            f = BOOST / (prod + '.csv.gz')
            if f.exists():
                with gzip.open(f, 'rt', encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        m[r['object_id']] = (r['boost_t1_class'], r['boost_t2_class'])
            per_prod_cache[prod] = m
        t1, t2 = per_prod_cache[prod].get(oid, ('', ''))
        y_t1.append(t1)
        y_t2.append(t2)

    res = {'fold': a.fold, 'run': str(a.run), 'vocabulary': vocab, 'B0': b0, 'retained': ret,
           'mapping_version': MAPPING_VERSION,
           'predictions_hash': hashlib.sha256(probs.tobytes()).hexdigest()[:16],
           'note': 'single-column auxiliary evaluation on boost labels; never used for training or model '
                   'selection; boost tier is NOT the frozen AIS-match label'}
    for tier, ys in (('t1', y_t1), ('t2', y_t2)):
        if tier == 't2':
            ys = [x if x else t for x, t in zip(y_t1, y_t2)]     # t1 preferred, t2 fills in
        keep = np.array([k for k, v in enumerate(ys) if v in ci])
        if not len(keep):
            res[tier] = {'n': 0}
            continue
        y = np.array([ci[ys[k]] for k in keep])
        block = {'n': int(len(keep)), 'coverage_of_eval': float(len(keep) / max(1, len(eidx)))}
        for name, heads in [('B0', [b0]), ('B1', list(range(probs.shape[0]))), ('Full', ret)]:
            p = probs[heads][:, keep, :].mean(0)
            pred = p.argmax(-1)
            block[name] = {'acc': float((pred == y).mean()), 'ba': balanced_accuracy(y, pred),
                           'per_class': {vocab[c]: {'n': int((y == c).sum()),
                                                    'recall': float((pred[y == c] == c).mean())} for c in sorted(set(y))}}
        # integrity: agreement with frozen fine labels where both exist
        both = [(frozen[ids[eidx[k]]][0], ys[k]) for k in keep
                if frozen.get(ids[eidx[k]], ('', ''))[0] and frozen[ids[eidx[k]]][1] == 'fine']
        if both:
            def coarse(c):
                if c in ('container_ship', 'general_cargo', 'bulk_carrier', 'ro_ro_vehicle_carrier',
                         'refrigerated_cargo', 'cargo_coarse'):
                    return 'cargo'
                if c in ('product_chemical_tanker', 'crude_oil_tanker', 'lpg_lng_tanker', 'bunker_tanker',
                         'tanker_coarse'):
                    return 'tanker'
                if c in ('tug_towing', 'pilot_port_tender'):
                    return 'tug'
                return c
            block['agreement_with_frozen'] = {
                'n': len(both),
                'exact': float(np.mean([a1 == b1 for a1, b1 in both])),
                'coarse': float(np.mean([coarse(a1) == coarse(b1) for a1, b1 in both]))}
        res[tier] = block

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('boost_eval_%s.json' % a.fold)).write_text(json.dumps(res, indent=1))
    for tier in ('t1', 't2'):
        b = res.get(tier, {})
        if b.get('n'):
            print('%-12s %s n=%4d  B0 %.4f/%.4f  B1 %.4f/%.4f  Full %.4f/%.4f' % (
                a.fold, tier, b['n'], b['B0']['acc'], b['B0']['ba'], b['B1']['acc'], b['B1']['ba'],
                b['Full']['acc'], b['Full']['ba']))
        else:
            print('%-12s %s n=0' % (a.fold, tier))


if __name__ == '__main__':
    main()

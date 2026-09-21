"""E01 boost-label evaluation — revision 2: adds the COARSE tier approved as option (A).

Evaluation-only change: the model, heads, relations and filter rules are untouched. The true
labels (boost t1/t2) and the fine argmax prediction are both projected onto coarse parents, and
every eval instance whose coarse parent is predictable by the fold's vocabulary becomes evaluable.

usage: python e01_boost_eval.py --fold Shanghai --run /root/autodl-tmp/e01_runs/Shanghai_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
BOOST = ROOT / 'e01_label_boost'
OUT = BOOST / 'eval'
MAPPING_VERSION = 'coarse_parent_v1'
COARSE = {
    'container_ship': 'cargo', 'general_cargo': 'cargo', 'bulk_carrier': 'cargo',
    'ro_ro_vehicle_carrier': 'cargo', 'refrigerated_cargo': 'cargo', 'cargo_coarse': 'cargo',
    'product_chemical_tanker': 'tanker', 'crude_oil_tanker': 'tanker', 'lpg_lng_tanker': 'tanker',
    'bunker_tanker': 'tanker', 'tanker_coarse': 'tanker',
    'passenger_ship': 'passenger', 'ro_ro_passenger': 'passenger', 'high_speed_passenger': 'passenger',
    'fishing_vessel': 'fishing', 'tug_towing': 'tug', 'pilot_port_tender': 'tug',
    'dredger': 'dredger', 'offshore_supply': 'offshore', 'offshore_vessel': 'offshore',
    'sailing_vessel': 'pleasure', 'pleasure_craft': 'pleasure', 'yacht': 'pleasure',
}


def coarse(c: str) -> str:
    return COARSE.get(c, 'other') if c else ''


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
    vocab_coarse = sorted({coarse(c) for c in vocab})
    probs = np.load(a.run / 'eval_probabilities.npy')
    eidx = np.load(a.run / 'eval_index.npy')
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    b0 = json.loads((a.run / 'meta_head_selection.json').read_text())['B0']
    ret = json.loads((a.run / 'adapt_filter' / 'retention.json').read_text())['retained']
    frozen = load_frozen()

    cache = {}
    y_t1, y_t2 = [], []
    for i in eidx:
        oid = ids[i]
        prod = oid.split('|')[0] if '|' in oid else ''
        if prod not in cache:
            m = {}
            f = BOOST / (prod + '.csv.gz')
            if f.exists():
                with gzip.open(f, 'rt', encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        m[r['object_id']] = (r['boost_t1_class'], r['boost_t2_class'])
            cache[prod] = m
        t1, t2 = cache[prod].get(oid, ('', ''))
        y_t1.append(t1)
        y_t2.append(t2)
    y_t2f = [a1 if a1 else a2 for a1, a2 in zip(y_t1, y_t2)]          # t1 preferred, t2 fills in

    res = {'fold': a.fold, 'run': str(a.run), 'vocabulary': vocab, 'vocabulary_coarse': vocab_coarse,
           'B0': b0, 'retained': ret, 'mapping_version': MAPPING_VERSION,
           'predictions_hash': hashlib.sha256(probs.tobytes()).hexdigest()[:16],
           'note': 'auxiliary evaluation only (boost labels + coarse projection); never used for training '
                   'or model selection; option (A) approved by the reviewer'}

    def report(tier, ys_fine, project_coarse: bool):
        if project_coarse:
            ys = [coarse(v) for v in ys_fine]
            allowed = set(vocab_coarse)
        else:
            ys = list(ys_fine)
            allowed = set(vocab)
        keep = np.array([k for k, v in enumerate(ys) if v in allowed])
        if not len(keep):
            return {'n': 0, 'coverage_of_eval': 0.0}
        y = np.array([(vocab_coarse.index(ys[k]) if project_coarse else ci[ys[k]]) for k in keep])
        labels = vocab_coarse if project_coarse else vocab
        block = {'n': int(len(keep)), 'coverage_of_eval': float(len(keep) / max(1, len(eidx))),
                 'classes': labels}
        preds = {}
        for name, heads in [('B0', [b0]), ('B1', list(range(probs.shape[0]))), ('Full', ret)]:
            p = probs[heads][:, keep, :].mean(0)
            fine_pred = p.argmax(-1)
            pred = np.array([labels.index(coarse(vocab[c])) if project_coarse else fine_pred[j]
                             for j, c in enumerate(fine_pred)])
            preds[name] = pred
            block[name] = {'acc': float((pred == y).mean()), 'ba': balanced_accuracy(y, pred),
                           'per_class': {labels[c]: {'n': int((y == c).sum()),
                                                     'recall': float((pred[y == c] == c).mean())} for c in sorted(set(y))}}
        p1, pF = preds['B1'], preds['Full']
        block['rescue_vs_B1'] = int(((p1 != y) & (pF == y)).sum())
        block['harm_vs_B1'] = int(((p1 == y) & (pF != y)).sum())
        both = [(frozen[ids[eidx[k]]][0], ys_fine[k]) for k in keep
                if frozen.get(ids[eidx[k]], ('', ''))[0] and frozen[ids[eidx[k]]][1] == 'fine']
        if both and not project_coarse:
            block['agreement_with_frozen'] = {
                'n': len(both),
                'exact': float(np.mean([a1 == b1 for a1, b1 in both])),
                'coarse': float(np.mean([coarse(a1) == coarse(b1) for a1, b1 in both]))}
        return block

    res['fine_t1'] = report('fine_t1', y_t1, False)
    res['fine_t2'] = report('fine_t2', y_t2f, False)
    res['coarse_t1'] = report('coarse_t1', y_t1, True)
    res['coarse_t2'] = report('coarse_t2', y_t2f, True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('boost_eval_%s.json' % a.fold)).write_text(json.dumps(res, indent=1))
    for tier in ('fine_t1', 'fine_t2', 'coarse_t1', 'coarse_t2'):
        b = res[tier]
        if b.get('n'):
            print('%-12s %-9s n=%5d  B0 %.4f/%.4f  B1 %.4f/%.4f  Full %.4f/%.4f  (rescue %d harm %d)'
                  % (a.fold, tier, b['n'], b['B0']['acc'], b['B0']['ba'], b['B1']['acc'], b['B1']['ba'],
                     b['Full']['acc'], b['Full']['ba'], b['rescue_vs_B1'], b['harm_vs_B1']))
        else:
            print('%-12s %-9s n=0' % (a.fold, tier))


if __name__ == '__main__':
    main()

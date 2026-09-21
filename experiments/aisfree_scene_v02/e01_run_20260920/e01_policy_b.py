"""(b) Decision-granularity policy, v2 — policies developed on SOURCE only, evaluated on target.

Two output spaces, each with its own source-trained policy set:
  fine    : the fold's fine vocabulary, truth = frozen fine labels on the target eval products
  coarse  : parent classes of that vocabulary (probability parent-sum, argmax after summing),
            truth = boost_v2 labels projected to parents

Policies (all fitted on SOURCE data of the same space, no target truth):
  B0   the single head chosen by the frozen pipeline (cross-port source BA)
  B1   equal-weight mean of all 8 heads
  FIX  best fixed subset chosen on source
  STK  per-object stacking: logistic regression on the concatenated head probability vectors
  OBJ  per-object best over the 255-subset family  -- ORACLE_ONLY indicator, not deployable

usage: python e01_policy_b.py --fold Shanghai --run /root/autodl-tmp/e01_runs/Shanghai_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
BOOST = ROOT / 'e01_label_boost_v2'
OUT = ROOT / 'e01_runs' / 'aux_policy_b'
PARENT = {
    'container_ship': 'cargo', 'general_cargo': 'cargo', 'bulk_carrier': 'cargo',
    'ro_ro_vehicle_carrier': 'cargo', 'refrigerated_cargo': 'cargo', 'cargo_coarse': 'cargo',
    'product_chemical_tanker': 'tanker', 'crude_oil_tanker': 'tanker', 'lpg_lng_tanker': 'tanker',
    'bunker_tanker': 'tanker', 'tanker_coarse': 'tanker',
    'passenger_ship': 'passenger', 'ro_ro_passenger': 'passenger', 'high_speed_passenger': 'passenger',
    'fishing_vessel': 'fishing', 'tug_towing': 'tug', 'pilot_port_tender': 'tug',
    'dredger': 'dredger', 'offshore_supply': 'offshore', 'offshore_vessel': 'offshore',
    'sailing_vessel': 'pleasure', 'pleasure_craft': 'pleasure', 'yacht': 'pleasure',
}


def ba(y, pred, n_classes):
    rs = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def acc(y, pred):
    return float((pred == y).mean()) if len(y) else float('nan')


def parent_sum(p, names):
    parents = []
    for c in names:
        a = PARENT.get(c)
        if a and a not in parents:
            parents.append(a)
    out = np.zeros((p.shape[0], p.shape[1], len(parents)))
    for ci, c in enumerate(names):
        a = PARENT.get(c)
        if a:
            out[:, :, parents.index(a)] += p[:, :, ci]
    return out, parents


def subset_mean(p, bits):
    hs = [h for h in range(p.shape[0]) if bits >> h & 1] or list(range(p.shape[0]))
    return p[hs].mean(0)


def best_fixed(p, y):
    best, best_a = None, -1.0
    for bits in range(1, 1 << p.shape[0]):
        a = acc(y, subset_mean(p, bits).argmax(-1))
        if a > best_a + 1e-12:
            best, best_a = bits, a
    return best, best_a


def stack_fit(p, y):
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([p[h] for h in range(p.shape[0])], 1)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced')
    clf.fit(X, y)
    return clf


def stack_pred(clf, p):
    return clf.predict(np.concatenate([p[h] for h in range(p.shape[0])], 1))


def obj_oracle_hits(p, y):
    hits = np.zeros(len(y), dtype=bool)
    for bits in range(1, 1 << p.shape[0]):
        hits |= (subset_mean(p, bits).argmax(-1) == y)
    return hits


def load_boost(products):
    lab = {}
    for pr in products:
        f = BOOST / (pr + '.csv.gz')
        if not f.exists():
            continue
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                lab[r['object_id']] = (r['v2_t1_class'], r['v2_t2_class'])
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=KS / 'e01_first_batch' / 'split_manifest.json')
    a = ap.parse_args()
    tag = a.fold.replace(' ', '')

    s = np.load(SRCDEV / (tag + '_source.npz'), allow_pickle=False)
    p_src = s['p'].astype(np.float64)
    y_src_f = s['y']
    vocab = [str(x) for x in s['class_names']]
    p_src_c, coarse = parent_sum(p_src, vocab)
    y_src_c = np.array([coarse.index(PARENT[vocab[i]]) for i in y_src_f])

    p_t = np.load(a.run / 'eval_probabilities.npy').astype(np.float64)
    p_t_c, _ = parent_sum(p_t, vocab)
    eidx = np.load(a.run / 'eval_index.npy')
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    b0 = int(json.loads((a.run / 'meta_head_selection.json').read_text())['B0'])
    del p_t, feats

    fold = json.loads(a.manifest.read_text())['folds'][a.fold]
    boost = load_boost(sorted(set(fold['target_eval_products'])))
    frozen = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['product_id'] in set(fold['target_eval_products']):
                frozen[r['object_id']] = (r.get('ais_final_class') or '').strip()

    spaces = {'fine': {'vocab': vocab, 'p_src': p_src, 'y_src': y_src_f, 'p_t': None, 'truth': 'frozen'},
              'coarse': {'vocab': coarse, 'p_src': p_src_c, 'y_src': y_src_c, 'p_t': None, 'truth': 'boost'}}

    # rebuild the target probabilities per space (fine was deleted to save RAM; reload once per space)
    p_fine = np.load(a.run / 'eval_probabilities.npy').astype(np.float64)
    p_coarse, _ = parent_sum(p_fine, vocab)
    spaces['fine']['p_t'], spaces['coarse']['p_t'] = p_fine, p_coarse

    res = {'fold': a.fold, 'H': int(p_src.shape[0]), 'B0_head': b0, 'views': {}}
    for sp_name, sp in spaces.items():
        names = sp['vocab']
        C = len(names)
        fix_bits, fix_a = best_fixed(sp['p_src'], sp['y_src'])
        clf = stack_fit(sp['p_src'], sp['y_src'])
        src = {'n': int(len(sp['y_src'])), 'B1': acc(sp['y_src'], sp['p_src'].mean(0).argmax(-1)),
               'FIX': fix_a, 'FIX_size': int(bin(fix_bits).count('1')),
               'STK': acc(sp['y_src'], stack_pred(clf, sp['p_src']))}
        # target view
        p_t = sp['p_t']
        vindex = {c: i for i, c in enumerate(names)}
        keep, y = [], []
        for k, i in enumerate(eidx):
            oid = ids[i]
            if sp['truth'] == 'frozen':
                lab = frozen.get(oid, '')
            else:
                r = boost.get(oid)
                if r is None:
                    continue
                lab = r[0] or r[1]
                lab = PARENT.get(lab, '')
            if lab in vindex:
                keep.append(k); y.append(vindex[lab])
        keep = np.array(keep, dtype=np.int64); y = np.array(y, dtype=np.int64)
        view = {'n': int(len(keep)), 'n_classes': int(len(set(y.tolist()))), 'source': src,
                'fixed_subset_bits': int(fix_bits), 'fixed_subset_size': int(bin(fix_bits).count('1'))}
        if len(keep):
            pt = p_t[:, keep, :]
            preds = {'B0': pt[b0].argmax(-1), 'B1': pt.mean(0).argmax(-1),
                     'FIX': subset_mean(pt, fix_bits).argmax(-1), 'STK': stack_pred(clf, pt)}
            hits = obj_oracle_hits(pt, y)
            view['policies'] = {k: {'acc': acc(y, v), 'ba': ba(y, v, C)} for k, v in preds.items()}
            view['OBJ_oracle'] = {'acc': float(hits.mean()), 'deployable': False}
            view['B1_errors'] = int((preds['B1'] != y).sum())
            view['B1_errors_stk_fixes'] = int(((preds['B1'] != y) & (preds['STK'] == y)).sum())
            view['B1_errors_stk_breaks'] = int(((preds['B1'] == y) & (preds['STK'] != y)).sum())
            del pt
        res['views'][sp_name] = view
        del sp['p_t']

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (tag + '_policy_b.json')).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for sp_name, v in res['views'].items():
        if not v['n']:
            print('%-10s %-7s n=0' % (a.fold, sp_name)); continue
        print('%-10s %-7s n=%4d(B1错%d) | ' % (a.fold, sp_name, v['n'], v['B1_errors']) +
              ' '.join('%s %.3f/%.3f' % (k, d['acc'], d['ba']) for k, d in v['policies'].items()) +
              ' OBJ %.3f | src x=%.3f FIX %.3f(%d) STK %.3f' % (
                  v['OBJ_oracle']['acc'], v['source']['B1'], v['source']['FIX'],
                  v['source']['FIX_size'], v['source']['STK']))


if __name__ == '__main__':
    main()

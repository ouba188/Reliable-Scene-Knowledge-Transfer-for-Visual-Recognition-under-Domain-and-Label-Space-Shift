"""Build the evaluator-side NPZ inputs for the reviewer's oracle_audit.py, then summarise.

Views per fold: fine_t1, fine_t1plusT2, coarse_t1, coarse_t1plusT2.
Coarse probabilities are parent SUMS (p_coarse[a] = sum of p_fine over children), argmax after summing.
Truth comes from boost_v2 labels; instances whose class is outside the view's output space are
recorded as coverage only and never forced into a class.

usage: python e01_prep_ceiling.py --fold Shanghai --run /root/autodl-tmp/e01_runs/Shanghai_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
BOOST = ROOT / 'e01_label_boost_v2'
PKG = KS / 'e01_first_batch'
OUT = ROOT / 'e01_runs' / 'aux_candidate_ceiling_v1'
TOOL = KS / 'e01_candidate_ceiling_v1' / 'oracle_audit.py'
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
VIEWS = ['fine_t1', 'fine_t1plusT2', 'coarse_t1', 'coarse_t1plusT2']


def parent_sum(p: np.ndarray, fine_names: list[str]):
    """p[H,N,Cf] -> p[H,N,Cp] by summing children per parent; parents keep first-appearance order."""
    parents = []
    for c in fine_names:
        a = PARENT.get(c)
        if a is None:
            continue
        if a not in parents:
            parents.append(a)
    out = np.zeros((p.shape[0], p.shape[1], len(parents)), dtype=np.float64)
    for ci, c in enumerate(fine_names):
        a = PARENT.get(c)
        if a is None:
            continue
        out[:, :, parents.index(a)] += p[:, :, ci]
    return out, parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()

    vocab = json.loads((a.run / 'heads' / 'class_vocab.json').read_text())
    probs = np.load(a.run / 'eval_probabilities.npy').astype(np.float64)     # [H,N,Cf]
    eidx = np.load(a.run / 'eval_index.npy')
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    ret = json.loads((a.run / 'adapt_filter' / 'retention.json').read_text())
    score = ret.get('score') or [None] * probs.shape[0]

    cache = {}
    y_t1, y_t2, prods = [], [], []
    for i in eidx:
        oid = ids[i]
        prod = oid.split('|')[0] if '|' in oid else ''
        if prod not in cache:
            m = {}
            f = BOOST / (prod + '.csv.gz')
            if f.exists():
                with gzip.open(f, 'rt', encoding='utf-8') as fh:
                    for r in csv.DictReader(fh):
                        m[r['object_id']] = (r['v2_t1_class'], r['v2_t2_class'])
            cache[prod] = m
        t1, t2 = cache[prod].get(oid, ('', ''))
        y_t1.append(t1); y_t2.append(t2); prods.append(prod)

    p_fine = probs
    p_coarse, coarse_names = parent_sum(p_fine, vocab)
    scores = np.array([np.inf if s is None else float(s) for s in score], dtype=np.float64)

    base = OUT / a.fold.replace(' ', '')
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for view in VIEWS:
        is_coarse = view.startswith('coarse')
        plus = view.endswith('plusT2')
        names = coarse_names if is_coarse else list(vocab)
        p = p_coarse if is_coarse else p_fine
        labels = [((PARENT.get(t1) if is_coarse else t1) if t1 else
                   ((PARENT.get(t2) if is_coarse else t2) if (plus and t2) else '')) for t1, t2 in zip(y_t1, y_t2)]
        idx_map = {n: k for k, n in enumerate(names)}
        keep = np.array([k for k, lab in enumerate(labels) if lab in idx_map], dtype=np.int64)
        kset = set(keep.tolist())
        coverage = {'view': view, 'eval_instances': int(len(eidx)), 'evaluable': int(len(keep)),
                    'coverage_fraction': float(len(keep) / max(1, len(eidx))),
                    'unlabeled': int(sum(1 for k in range(len(labels)) if k not in kset and labels[k] == '')),
                    'out_of_output_space': int(sum(1 for k in range(len(labels))
                                                  if labels[k] and k not in kset))}
        if not len(keep):
            rows.append({**coverage, 'n': 0})
            continue
        y = np.array([idx_map[labels[k]] for k in keep], dtype=np.int64)
        out_dir = base / view
        npz = base / ('input_%s.npz' % view)
        blob = {'p': p[:, keep, :].astype(np.float32), 'y': y,
                'sample_id': ids[eidx[keep]].astype('<U128'),
                'class_names': np.array(names, dtype='<U64'),
                'product_id': np.array([prods[k] for k in keep], dtype='<U128'),
                'b1_pred': p[:, keep, :].mean(0).argmax(-1).astype(np.int64)}
        # score is optional in the reviewer interface: omit it entirely when any head lacks a finite
        # frozen adapt score (never substitute a fabricated value); the score-prefix bound is then reported null.
        if np.all(np.isfinite(scores)) and len(scores) == probs.shape[0]:
            blob['score'] = scores.astype(np.float32)
        coverage['score_available'] = bool('score' in blob)
        np.savez_compressed(npz, **blob)
        r = subprocess.run([sys.executable, str(TOOL), '--input', str(npz), '--view', view,
                            '--out', str(out_dir)], capture_output=True, text=True)
        rep = out_dir / 'ORACLE_ONLY_report.json'
        summary = {}
        if rep.exists():
            d = json.loads(rep.read_text())
            b1 = d.get('B1', {})
            fsa = d.get('fixed_subset_ORACLE_acc', {})
            fsb = d.get('fixed_subset_ORACLE_ba', {})
            sha = d.get('single_head_ORACLE_acc', {})
            sp = d.get('score_prefix_ORACLE', {})
            sw = d.get('samplewise_subset_ORACLE', {})
            summary = {'n': coverage['evaluable'], 'b1_acc': b1.get('acc'), 'b1_ba': b1.get('ba'),
                       'b1_errors': d.get('B1_error_count'),
                       'head_disagreement': d.get('mean_pairwise_disagreement'),
                       'best_single_head_acc': sha.get('acc'), 'best_single_head_ba': sha.get('ba'),
                       'best_single_head_mask': sha.get('subset_mask'),
                       'fixed_subset_acc': fsa.get('acc'), 'fixed_subset_ba': fsb.get('ba'),
                       'fixed_subset_mask_acc': fsa.get('subset_mask'), 'fixed_subset_size': fsa.get('size'),
                       'score_prefix_acc': sp.get('acc_best', {}).get('acc') if isinstance(sp.get('acc_best'), dict) else sp.get('acc_best'),
                       'score_prefix_ba': sp.get('ba_best', {}).get('ba') if isinstance(sp.get('ba_best'), dict) else sp.get('ba_best'),
                       'samplewise_acc': sw.get('acc'), 'samplewise_ba': sw.get('ba'),
                       'b1_prediction_verified': d.get('original_B1_prediction_verified'),
                       'n_distinct_products': d.get('n_distinct_products')}
        else:
            summary = {'n': coverage['evaluable'], 'error': (r.stderr or 'no report')[-300:]}
        rows.append({**coverage, **summary})
        print('%-6s %-16s evaluable %5d  %s' % (a.fold, view, coverage['evaluable'],
                                                json.dumps(summary, ensure_ascii=False)[:180]))

    (base / 'ceiling_summary.json').write_text(json.dumps({'fold': a.fold, 'views': rows}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

"""E01 P3 attribution: Static(D0-D2) / Group(D3) / Full ablations + B3/B4/B5 baselines.

All variants reuse the frozen per-fold artifacts (probabilities, groups, phi/available, relation
means, calibration tau). No training, no rule changes, no target labels inside any model input.

  Static : aggregate with only D0-D2 enabled
  Group  : aggregate with only D3 enabled
  Full   : aggregate with the fold's enabled dimensions (as stored)
  B3     : unconditional relation moments (single pooled group)
  B4     : conditional moments, then keep only the single best head
  B5     : per-group mechanism re-selection (each group picks its own arg-min mechanism)

usage: python e01_p3_attribution.py --fold Rotterdam --run /root/autodl-tmp/e01_runs/Rotterdam_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
PKG = KS / 'e01_first_batch'
OUT = ROOT / 'e01_runs' / 'aux_p3'
sys.path.insert(0, str(PKG / 'tools'))
from e01_core import aggregate, filter_bank  # noqa: E402

COARSE = {
    'container_ship': 'cargo', 'general_cargo': 'cargo', 'bulk_carrier': 'cargo',
    'ro_ro_vehicle_carrier': 'cargo', 'refrigerated_cargo': 'cargo', 'cargo_coarse': 'cargo',
    'product_chemical_tanker': 'tanker', 'crude_oil_tanker': 'tanker', 'lpg_lng_tanker': 'tanker',
    'bunker_tanker': 'tanker', 'tanker_coarse': 'tanker',
    'passenger_ship': 'passenger', 'ro_ro_passenger': 'passenger', 'high_speed_passenger': 'passenger',
    'fishing_vessel': 'fishing', 'tug_towing': 'tug', 'pilot_port_tender': 'tug',
    'dredger': 'dredger', 'offshore_supply': 'offshore', 'offshore_vessel': 'offshore',
    'sailing_vessel': 'pleasure', 'pleasure_craft': 'pleasure', 'research_vessel': 'other',
}


def load_meta():
    meta = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                meta[r['object_id']] = r
    return meta


def eval_metrics(probs, heads, eidx, ids, meta, vocab):
    ci = {c: k for k, c in enumerate(vocab)}
    y_fine = [meta.get(ids[i], {}).get('ais_final_class', '') for i in eidx]
    keep = np.array([k for k, v in enumerate(y_fine) if v in ci])
    if not len(keep):
        return {'n': 0}
    y = np.array([ci[y_fine[k]] for k in keep])
    p = probs[heads][:, keep, :].mean(0)
    pred = p.argmax(-1)
    acc = float((pred == y).mean())
    rec = {}
    for c in sorted(set(y)):
        sel = y == c
        rec[vocab[c]] = {'n': int(sel.sum()), 'recall': float((pred[sel] == c).mean())}
    ba = float(np.mean([v['recall'] for v in rec.values()]))
    return {'n': int(len(y)), 'acc': acc, 'ba': ba, 'per_class': rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=PKG / 'config.json')
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    run = a.run
    fi = np.load(run / 'filter_inputs.npz', allow_pickle=False)
    phi, av = fi['phi'].astype(np.float64), fi['available'].astype(bool)
    grp, blocks = fi['groups'], fi['spatial_blocks']
    probs, q = fi['probabilities'].astype(np.float64), fi['relation_means'].astype(np.float64)
    enabled = fi['enabled_dimensions'].astype(bool) if 'enabled_dimensions' in fi.files else np.ones(4, bool)
    cal = json.loads((run / 'calibration.json').read_text())
    tau = cal.get('tau')
    b0 = json.loads((run / 'meta_head_selection.json').read_text())['B0']
    vocab = json.loads((run / 'heads' / 'class_vocab.json').read_text())
    feats = np.load(run / 'features.npz', allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    eidx = np.load(run / 'eval_index.npy')
    # evaluation probabilities live in their own file: [H, N_eval, C]; keep indexes positions there
    probs_eval = np.load(run / 'eval_probabilities.npy').astype(np.float64)
    meta = load_meta()
    min_cell = cfg['moment_validity']['min_available_objects_per_cell']
    min_blocks = cfg['moment_validity']['min_spatial_blocks_per_cell']

    def run_variant(tag, en, groups=None, pooled=False, **kw):
        g = np.zeros(len(grp), np.int64) if pooled else np.asarray(groups if groups is not None else grp)
        m = aggregate(phi, av, g, blocks, probs, q, B=cfg['B'], min_cell=min_cell,
                      min_blocks=min_blocks, enabled_dimensions=en)
        if tau is None:
            r = {'retained': list(range(probs.shape[0])), 'status': 'calibration_unavailable'}
        else:
            r = filter_bank(m.observed, m.predicted, m.active, float(tau))
        active = int(m.active.sum())
        out = {'variant': tag, 'active_cells': active,
               'status': r.get('status'), 'retained': r.get('retained'),
               'per_head_score': [None if s is None else round(float(s), 4) for s in r.get('score', [])]}
        out['eval'] = eval_metrics(probs_eval, r['retained'], eidx, ids, meta, vocab)
        out['eval_B0'] = eval_metrics(probs_eval, [b0], eidx, ids, meta, vocab)
        out['eval_B1'] = eval_metrics(probs_eval, list(range(probs_eval.shape[0])), eidx, ids, meta, vocab)
        return out, m, r

    variants = []
    static = enabled.copy(); static[3] = False
    group_only = np.zeros(4, bool); group_only[3] = True
    for tag, en in [('Static_D0-D2', static), ('Group_D3', group_only), ('Full', enabled)]:
        out, _m, _r = run_variant(tag, en)
        variants.append(out)
    # B3: pooled single group
    out, _m, _r = run_variant('B3_unconditional_relation', enabled, pooled=True)
    variants.append(out)
    # B4: conditional moments, keep only the single best head
    m4 = aggregate(phi, av, grp, blocks, probs, q, B=cfg['B'], min_cell=min_cell,
                   min_blocks=min_blocks, enabled_dimensions=enabled)
    r4 = filter_bank(m4.observed, m4.predicted, m4.active, float(tau)) if tau is not None else {'retained': [b0], 'score': [None] * probs.shape[0]}
    scores = np.array([np.inf if s is None else s for s in r4.get('score', [np.inf] * probs.shape[0])])
    best = int(np.argmin(scores))
    variants.append({'variant': 'B4_single_best_interpretation', 'active_cells': int(m4.active.sum()),
                     'status': 'single_head', 'retained': [best], 'per_head_score': r4.get('score'),
                     'eval': eval_metrics(probs_eval, [best], eidx, ids, meta, vocab),
                     'eval_B0': eval_metrics(probs_eval, [b0], eidx, ids, meta, vocab),
                     'eval_B1': eval_metrics(probs_eval, list(range(probs_eval.shape[0])), eidx, ids, meta, vocab)})
    # B5: per-group mechanism re-selection (each group picks its own arg-min mechanism)
    sel = []
    for b in range(cfg['B']):
        act = m4.active
        sc = []
        for j in range(q.shape[0]):
            res = []
            for d in range(phi.shape[1]):
                if act[b, d]:
                    res.append(float(np.nanmax(np.abs(m4.predicted[:, j, b, d] - m4.observed[b, d]))))
            sc.append(max(res) if res else np.inf)
        sel.append(int(np.argmin(sc)))
    variants.append({'variant': 'B5_per_group_mechanism', 'active_cells': int(m4.active.sum()),
                     'status': 'per_group_mechanism', 'retained': r4.get('retained'),
                     'group_mechanism': sel, 'per_head_score': r4.get('score'),
                     'eval': eval_metrics(probs_eval, r4.get('retained', list(range(probs_eval.shape[0]))), eidx, ids, meta, vocab),
                     'eval_B0': eval_metrics(probs_eval, [b0], eidx, ids, meta, vocab),
                     'eval_B1': eval_metrics(probs_eval, list(range(probs_eval.shape[0])), eidx, ids, meta, vocab)})

    rec = {'fold': a.fold, 'run': str(run), 'tau': tau, 'calibration_status': cal.get('status'),
           'B0': b0, 'enabled_dimensions': enabled.tolist(), 'variants': variants,
           'note': 'attribution only; no training and no rule changes'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('p3_%s.json' % a.fold)).write_text(json.dumps(rec, indent=1))
    print('P3 %s (tau=%s, enabled=%s)' % (a.fold, tau, enabled.tolist()))
    for v in variants:
        e = v['eval']
        print('  %-28s active=%2d retained=%-16s n=%-4s acc=%s ba=%s' % (
            v['variant'], v['active_cells'], str(v['retained'])[:16], e.get('n'),
            None if e.get('n') == 0 else round(e['acc'], 4), None if e.get('n') == 0 else round(e['ba'], 4)))


if __name__ == '__main__':
    main()

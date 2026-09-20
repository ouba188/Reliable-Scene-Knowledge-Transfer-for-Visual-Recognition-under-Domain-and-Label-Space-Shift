"""E01 P1 mechanism analysis: relation-error decomposition, head disagreement, mechanism spread.

Reads the per-fold artifacts written by the pipeline (no training, no target labels in any model
input; target labels are only used by the evaluator-side disagreement report):

  run/adapt_filter/moments.npz   observed[B,D], predicted[H,J,B,D], active[B,D], counts, blocks, reasons
  run/heads/bank_manifest.json   per-head training info
  run/meta_head_selection.json   per-head meta BA + B0
  run/filter_inputs.npz          probabilities[H,N,C], relation_means[J,N,C,D], phi/available
  run/relations/relation_manifest.json   enabled_dimensions + source support

usage: python e01_p1_analysis.py --fold Rotterdam --run /root/autodl-tmp/e01_runs/Rotterdam_rev1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path('/root/autodl-tmp/e01_runs/aux_p1')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()

    run = a.run
    m = np.load(run / 'adapt_filter' / 'moments.npz', allow_pickle=False)
    obs, pred, act = m['observed'], m['predicted'], m['active']
    cnt, blk, rsn = m['counts'], m['blocks'], m['reasons']
    B, D = obs.shape
    H, J = pred.shape[:2]
    fi = np.load(run / 'filter_inputs.npz', allow_pickle=False)
    probs = fi['probabilities']          # [H,N,C]
    q = fi['relation_means']             # [J,N,C,D]
    phi, av = fi['phi'], fi['available']
    rman = json.loads((run / 'relations' / 'relation_manifest.json').read_text())
    heads = json.loads((run / 'heads' / 'bank_manifest.json').read_text())
    msel = json.loads((run / 'meta_head_selection.json').read_text())
    ret = json.loads((run / 'adapt_filter' / 'retention.json').read_text())

    # --- 1. relation-error decomposition per (b, d) ---
    cells = []
    for b in range(B):
        for d in range(D):
            p_here = pred[:, :, b, d]
            zero_src = float((np.abs(phi[..., d][av[..., d]]) < 1e-9).mean()) if av[..., d].any() else None
            cells.append({
                'group': b, 'dim': d, 'active': bool(act[b, d]), 'count': int(cnt[b, d]),
                'blocks': int(blk[b, d]), 'reason': str(rsn[b, d]),
                'observed': None if np.isnan(obs[b, d]) else float(obs[b, d]),
                'obs_zero_fraction': zero_src,
                'predicted_min': None if np.isnan(p_here).all() else float(np.nanmin(p_here)),
                'predicted_max': None if np.isnan(p_here).all() else float(np.nanmax(p_here)),
                'predicted_mean': None if np.isnan(p_here).all() else float(np.nanmean(p_here)),
                'spread_head': None if np.isnan(p_here).all() else float(np.nanmax(np.nanmax(p_here, 1) - np.nanmin(p_here, 1))),
                'spread_mech': None if np.isnan(p_here).all() else float(np.nanmax(np.nanmax(p_here, 0) - np.nanmin(p_here, 0))),
                'worst_abs_residual': None if np.isnan(p_here).all() or np.isnan(obs[b, d])
                else float(np.nanmax(np.abs(p_here - obs[b, d]))),
            })

    # --- 2. head disagreement on target adapt (argmax over the vocabulary) ---
    arg = probs.argmax(-1)                      # [H,N]
    dis = []
    for h1 in range(H):
        for h2 in range(h1 + 1, H):
            dis.append({'h1': h1, 'h2': h2, 'disagreement': float((arg[h1] != arg[h2]).mean())})
    # per-head confidence + ensemble agreement
    conf = probs.max(-1).mean(-1)
    unan = np.all(arg == arg[:1], axis=0).mean()

    # --- 3. mechanism spread per (j, d): mean |q_j - q_j'| over candidates ---
    mech = []
    for d in range(D):
        for j1 in range(J):
            for j2 in range(j1 + 1, J):
                mech.append({'dim': d, 'j1': j1, 'j2': j2,
                             'mean_abs_q_diff': float(np.abs(q[j1, :, :, d] - q[j2, :, :, d]).mean())})

    # --- 4. per-head meta BA + bank train loss sanity ---
    head_info = [{'head_id': r['head_id'], 'architecture': r['architecture'], 'sampling': r['sampling'],
                  'seed': r['seed'], 'last_loss': r['last_loss']} for r in heads['heads']]

    rec = {
        'fold': a.fold, 'run': str(run),
        'H': H, 'J': J, 'B': B, 'D': D,
        'enabled_dimensions': rman['enabled_dimensions'],
        'dimension_support': rman.get('support'),
        'B0': msel['B0'], 'per_head_meta_BA': msel.get('per_port_BA_mean') or msel.get('per_port_BA_by_head'),
        'retained': ret['retained'], 'filter_status': ret['status'],
        'relation_cells': cells,
        'head_pairwise_disagreement': dis,
        'mean_pairwise_disagreement': float(np.mean([d['disagreement'] for d in dis])) if dis else None,
        'unanimous_fraction': float(unan),
        'per_head_mean_confidence': conf.tolist(),
        'mechanism_spread': mech,
        'max_mechanism_spread': float(max(m['mean_abs_q_diff'] for m in mech)) if mech else None,
        'heads': head_info,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('p1_%s.json' % a.fold)).write_text(json.dumps(rec, indent=1))
    with (OUT / 'p1_relation_cells.csv').open('a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['fold'] + list(cells[0].keys()))
        for c in cells:
            w.writerow([a.fold] + [c[k] for k in cells[0]])
    print('P1 %s: 平均 H 分歧 %.4f, 全体一致比例 %.3f, 最大机制矩差 %.5f' % (
        a.fold, rec['mean_pairwise_disagreement'], rec['unanimous_fraction'], rec['max_mechanism_spread']))
    print('  活跃单元:', [(c['group'], c['dim']) for c in cells if c['active']])
    for c in cells:
        if not c['active']:
            print('   停用 (%d,%d): %s (count %d, blocks %d)' % (c['group'], c['dim'], c['reason'], c['count'], c['blocks']))


if __name__ == '__main__':
    main()

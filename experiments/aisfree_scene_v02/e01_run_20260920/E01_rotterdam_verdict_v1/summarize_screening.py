"""Read-only threshold geometry. Does NOT select or write a deployment threshold."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(retention: dict, margin: float = .02, floor: float = .05) -> dict:
    scores = np.asarray(retention['score'], dtype=float)
    cal = retention['calibration']
    vals = [(i, float(v)) for i, v in enumerate(cal['scores']) if v is not None]
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not vals:
        raise ValueError('Need finite per-head scores and at least one calibration score.')
    H = len(scores); tau = float(cal['tau'])
    v = np.array([x for _, x in vals])
    loo = []
    if len(v) > 1:
        for k, (i, _) in enumerate(vals):
            t = max(floor, float(np.delete(v, k).max()) + margin)
            loo.append({'omitted_calibration_index': i, 'diagnostic_tau': t,
                        'feasible_heads': int((scores <= t + 1e-12).sum())})
    min_bounded_quantile_tau = max(floor, float(v.min()) + margin)
    retained = retention['retained']
    return {
        'scope': 'Arithmetic diagnostics of saved scores; not validation of images/labels/moments.',
        'H': H, 'retained_count': len(retained),
        'all_heads_retained': sorted(retained) == list(range(H)),
        'score_min': float(scores.min()), 'score_max': float(scores.max()),
        'score_spread': float(np.ptp(scores)), 'tau': tau,
        'margin_above_max_head_score': tau - float(scores.max()),
        'minimum_calibration_score': float(v.min()),
        'minimum_possible_tau_from_bounded_quantile_plus_same_margin': min_bounded_quantile_tau,
        'every_quantile_between_cal_min_max_plus_same_margin_keeps_all': bool(min_bounded_quantile_tau >= scores.max()),
        'leave_one_cal_port_out_diagnostics': loo,
        'do_not_deploy_diagnostic_thresholds': True,
        'do_not_use_target_accuracy_to_select_tau': True,
        'pruning_intervals': {
            'tau_below_min_score': 'none feasible, then original rule falls back to all H',
            'tau_between_min_and_max': 'some heads may survive; endpoints follow <= and ties',
            'tau_at_or_above_max_score': 'all H survive'
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retention', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    r = summarize(json.loads(a.retention.read_text(encoding='utf8')))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(r, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()

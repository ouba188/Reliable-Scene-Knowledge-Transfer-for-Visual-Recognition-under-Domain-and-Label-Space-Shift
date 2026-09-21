"""E03 step B: leave-one-SOURCE-FOLD-out evaluation of a threshold rule on an unlabelled statistic.

Rule family: "apply mean-only iff stat >= t" (or <= t), threshold chosen on the OTHER folds' pairs only,
then applied to the held-out fold's pairs. Compared against always-cross, always-mean-only and the
per-pair oracle (which is not deployable).

usage: python e03_rule_eval.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
IN = ROOT / 'e03_pair_rule' / 'pair_diagnosis.csv'
OUT = ROOT / 'e03_pair_rule'


def main() -> None:
    rows = list(csv.DictReader(IN.open(encoding='utf-8')))
    for r in rows:
        for k in list(r):
            if k.startswith('stat_') or k.startswith('delta_') or k.endswith('_acc') or k.endswith('_ba'):
                r[k] = float(r[k])
    folds = sorted({r['fold'] for r in rows})
    stats = [k for k in rows[0] if k.startswith('stat_')]
    res = {}
    for s in stats:
        got, oracle, base_cross, base_mean, picks = [], [], [], [], []
        for f in folds:
            tr = [r for r in rows if r['fold'] != f]
            te = [r for r in rows if r['fold'] == f]
            vals = np.array([r[s] for r in tr]); dd = np.array([r['delta_acc'] for r in tr])
            best = None
            for direction in (1, -1):
                for t in np.unique(np.round(vals, 6)):
                    sel = (direction * vals) >= (direction * t)
                    if sel.sum() == 0 or sel.sum() == len(sel):
                        continue
                    gain = dd[sel].mean() * (sel.mean())      # accuracy gain over always-cross
                    if best is None or gain > best[0]:
                        best = (gain, direction, float(t))
            if best is None:
                best = (0.0, 1, float(np.median(vals)))
            _, direction, t = best
            for r in te:
                v = direction * r[s]
                use = v >= direction * t
                got.append(r['mean_acc'] if use else r['cross_acc'])
                oracle.append(max(r['mean_acc'], r['cross_acc']))
                base_cross.append(r['cross_acc']); base_mean.append(r['mean_acc'])
                picks.append(int(use))
        res[s] = {'rule_acc': float(np.mean(got)), 'always_cross': float(np.mean(base_cross)),
                  'always_mean': float(np.mean(base_mean)), 'oracle': float(np.mean(oracle)),
                  'picked_mean_only': int(np.sum(picks)), 'n': len(got)}
    order = sorted(res.items(), key=lambda kv: -(kv[1]['rule_acc'] - kv[1]['always_cross']))
    print('%-22s %10s %12s %12s %10s %8s' % ('统计量', '规则Acc', '总是cross', '总是mean', 'oracle', '选了mean'))
    for s, v in order:
        print('%-22s %10.3f %12.3f %12.3f %10.3f %8d' % (s, v['rule_acc'], v['always_cross'],
                                                         v['always_mean'], v['oracle'], v['picked_mean_only']))
    best = order[0]
    payload = {'evaluation': 'leave-one-source-fold-out', 'n_pairs': len(rows),
               'folds': folds, 'per_statistic': res, 'best_statistic': best[0],
               'best_gain_over_always_cross': best[1]['rule_acc'] - best[1]['always_cross'],
               'always_cross_acc': best[1]['always_cross'], 'oracle_acc': best[1]['oracle']}
    (OUT / 'rule_loo_eval.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print('\n最佳：%s  规则 %.3f vs 总是cross %.3f（+%.3f）；oracle %.3f' % (
        best[0], best[1]['rule_acc'], best[1]['always_cross'],
        best[1]['rule_acc'] - best[1]['always_cross'], best[1]['oracle']))


if __name__ == '__main__':
    main()

"""E03 step A: is the sign of mean-only's effect predictable from UNLABELLED statistics alone?

For every ordered source-port pair (A = reference pool, B = adaptation pool) inside a fold:
  statistics (deployment-legal, no labels): shift norm, energy distance proxy, classifier entropy
      on B before/after mean-only, mean top-1 confidence before/after
  outcome (labels, validation only): delta_acc = acc(mean_only) - acc(cross), delta_ba

Then: does any single statistic separate positive from negative pairs? Reported as AUC + Spearman,
per statistic, plus the source-leave-one-fold-out threshold rule's own triage accuracy.

usage: python e03_pair_rule.py [--min-port 150]
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
OUT = ROOT / 'e03_pair_rule'
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'Fujairah', 'JebelAli', 'PortSaid']
PCA_DIMS = 64


def ent(p):
    p = np.clip(p, 1e-9, 1)
    return float(-(p * np.log(p)).sum(1).mean())


def energy_distance(X, Y, cap=800, seed=20260920):
    """Unlabelled two-sample statistic: 2E|X-Y| - E|X-X'| - E|Y-Y'| (subsampled)."""
    rng = np.random.default_rng(seed)
    def sub(Z):
        return Z if len(Z) <= cap else Z[rng.choice(len(Z), cap, replace=False)]
    X, Y = sub(X), sub(Y)
    d_xy = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2).mean()
    d_xx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2).mean()
    d_yy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2).mean()
    return float(2 * d_xy - d_xx - d_yy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-port', type=int, default=150)
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    rows = []
    for fold in FOLDS:
        d = np.load(SRCDEV / (fold + '_source.npz'), allow_pickle=False)
        z_raw, y, port = d['z'].astype(np.float64), d['y'], d['port'].astype(str)
        C = len(set(y.tolist()))
        ports = [p for p in sorted(set(port.tolist())) if (port == p).sum() >= a.min_port]
        for A, B in permutations(ports, 2):
            Xa_raw, ya = z_raw[port == A], y[port == A]
            Xb_raw, yb = z_raw[port == B], y[port == B]
            if len(set(ya.tolist())) < 2 or len(set(yb.tolist())) < 2:
                continue
            pca = PCA(n_components=min(PCA_DIMS, Xa_raw.shape[1], max(2, Xa_raw.shape[0] - 1)),
                      random_state=20260920).fit(Xa_raw)
            Xa, Xb = pca.transform(Xa_raw), pca.transform(Xb_raw)
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xa, ya)
            cross = clf.predict(Xb)
            shift = Xb.mean(0) - Xa.mean(0)
            Xb_corr = Xb - Xb.mean(0) + Xa.mean(0)
            mean_pred = clf.predict(Xb_corr)
            pb, pbc = clf.predict_proba(Xb), clf.predict_proba(Xb_corr)
            rows.append({
                'fold': fold, 'ref_port': A, 'adapt_port': B, 'n_ref': int(len(Xa)), 'n_adapt': int(len(Xb)),
                'stat_shift_norm': float(np.linalg.norm(shift) / max(1e-9, Xa.std(0).mean())),
                'stat_energy_dist': energy_distance(Xa, Xb),
                'stat_entropy_before': ent(pb), 'stat_entropy_after': ent(pbc),
                'stat_delta_entropy': ent(pbc) - ent(pb),
                'stat_conf_before': float(pb.max(1).mean()), 'stat_conf_after': float(pbc.max(1).mean()),
                'stat_delta_conf': float(pbc.max(1).mean() - pb.max(1).mean()),
                'delta_acc': float((mean_pred == yb).mean() - (cross == yb).mean()),
                'delta_ba': None,
                'cross_acc': float((cross == yb).mean()), 'mean_acc': float((mean_pred == yb).mean()),
            })
            rec = rows[-1]
            def ba(pred):
                rs = [float((pred[yb == c] == c).mean()) for c in range(C) if (yb == c).any()]
                return float(np.mean(rs)) if rs else float('nan')
            rec['delta_ba'] = ba(mean_pred) - ba(cross)
            rec['cross_ba'], rec['mean_ba'] = ba(cross), ba(mean_pred)

    r = np.array([x['delta_acc'] for x in rows]); pos = r > 0
    stats = [k for k in rows[0] if k.startswith('stat_')]
    summary = {'n_pairs': len(rows), 'positive_pairs': int(pos.sum()),
               'delta_acc_mean': float(r.mean()), 'delta_acc_min': float(r.min()),
               'delta_acc_max': float(r.max()), 'statistics': {}}
    from scipy.stats import spearmanr
    for s in stats:
        v = np.array([x[s] for x in rows])
        try:
            rho, _ = spearmanr(v, r)
        except Exception:
            rho = float('nan')
        # AUC of the statistic as a classifier of "mean-only helps"
        order = np.argsort(v)
        ranks = np.empty(len(v)); ranks[order] = np.arange(len(v)) + 1
        n1, n0 = int(pos.sum()), int((~pos).sum())
        auc = float('nan') if n1 == 0 or n0 == 0 else float(
            (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        summary['statistics'][s] = {'spearman': float(rho), 'auc': auc,
                                    'auc_oriented': max(auc, 1 - auc) if auc == auc else auc}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'pair_diagnosis.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (OUT / 'pair_diagnosis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print('配对数 %d，mean-only 为正 %d 对；delta_acc 均值 %+.3f（%.3f ~ %+.3f）' % (
        summary['n_pairs'], summary['positive_pairs'], summary['delta_acc_mean'],
        summary['delta_acc_min'], summary['delta_acc_max']))
    print('%-22s %8s %8s %8s' % ('无标签统计量', 'Spearman', 'AUC', '定向AUC'))
    for s, sv in sorted(summary['statistics'].items(), key=lambda kv: -(kv[1]['auc_oriented'] or 0)):
        print('%-22s %8.3f %8.3f %8.3f' % (s, sv['spearman'], sv['auc'], sv['auc_oriented']))


if __name__ == '__main__':
    main()

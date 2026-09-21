"""(a) Domain-related separability: leave-one-source-port-out linear probe on the frozen features.

For each source port P of a fold, using only source instances (no target truth):
  cross   logistic probe trained on z of the OTHER source ports -> accuracy on P
  in      logistic probe with 5-fold CV inside P                -> accuracy on P (in-domain upper)
  coral   same as cross, but P's z is mean/covariance-aligned to the other ports first
          (CORAL-style whitening + re-colouring, unsupervised: uses target-side z only)
The gap in->cross quantifies domain shift; the coral column says how much of it a simple
second-order alignment recovers.

usage: python e01_domain_a.py --fold Rotterdam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
OUT = ROOT / 'e01_runs' / 'aux_domain_a'
MIN_PORT_N = 150


def fit_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(X, y)
    return clf


def cv_acc(X, y, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    accs = []
    skf = StratifiedKFold(n_splits=min(folds, max(2, int(np.bincount(y).min()))), shuffle=True,
                          random_state=20260920)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=3000, C=1.0)
        if len(set(y[tr].tolist())) < 2:
            continue
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float('nan')


def coral(src, tgt, eps=1e-6):
    """Align tgt to src by whitening+recolouring (unsupervised, uses tgt z only)."""
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    cs = np.cov(src, rowvar=False) + eps * np.eye(src.shape[1])
    ct = np.cov(tgt, rowvar=False) + eps * np.eye(tgt.shape[1])
    def whiten(c):
        w, v = np.linalg.eigh(c)
        w = np.clip(w, eps, None)
        return v @ np.diag(w ** -0.5) @ v.T
    W = whiten(cs) @ whiten(ct)
    return (tgt - mu_t) @ W + mu_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    a = ap.parse_args()
    tag = a.fold.replace(' ', '')
    d = np.load(SRCDEV / (tag + '_source.npz'), allow_pickle=False)
    z, y, port = d['z'].astype(np.float64), d['y'], d['port'].astype(str)
    vocab = [str(x) for x in d['class_names']]
    # ponytail: PCA to 64 dims before probing - 512-d logistic fits are ~10x slower and the
    # cross-port vs in-domain contrast (the quantity of interest) survives the projection.
    from sklearn.decomposition import PCA
    n_pc = min(64, z.shape[1], max(2, z.shape[0] - 1))
    z = PCA(n_components=n_pc, random_state=20260920).fit_transform(z)

    rows = []
    for P in sorted(set(port.tolist())):
        m_p = port == P
        m_o = ~m_p
        if m_p.sum() < MIN_PORT_N or len(set(y[m_p].tolist())) < 2:
            continue
        if len(set(y[m_o].tolist())) < 2:
            continue
        cross = fit_probe(z[m_o], y[m_o])
        a_cross = float((cross.predict(z[m_p]) == y[m_p]).mean())
        a_in = cv_acc(z[m_p], y[m_p])
        a_coral = float((cross.predict(coral(z[m_o], z[m_p])) == y[m_p]).mean())
        rows.append({'port': P, 'n': int(m_p.sum()), 'classes': len(set(y[m_p].tolist())),
                     'cross_acc': a_cross, 'in_acc': a_in, 'coral_acc': a_coral,
                     'gap_in_minus_cross': a_in - a_cross, 'coral_recovery': a_coral - a_cross})

    OUT.mkdir(parents=True, exist_ok=True)
    res = {'fold': a.fold, 'source_n': int(len(y)), 'ports': rows,
           'mean_gap': float(np.mean([r['gap_in_minus_cross'] for r in rows])) if rows else None,
           'mean_coral_recovery': float(np.mean([r['coral_recovery'] for r in rows])) if rows else None}
    (OUT / (tag + '_domain_a.json')).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for r in rows:
        print('%-10s %-22s n=%4d C=%d  cross %.3f  in %.3f  coral %.3f  gap %+.3f  恢复 %+.3f' % (
            a.fold, r['port'][:22], r['n'], r['classes'], r['cross_acc'], r['in_acc'],
            r['coral_acc'], r['gap_in_minus_cross'], r['coral_recovery']))
    print('%-10s 平均: 域间隙 %+.3f  CORAL 恢复 %+.3f（%d 个源港）' % (
        a.fold, res['mean_gap'] or float('nan'), res['mean_coral_recovery'] or float('nan'), len(rows)))


if __name__ == '__main__':
    main()

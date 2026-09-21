"""E02 step 1: CORAL numeric regression on the existing source caches + role lineage.

Fixes ONLY the CORAL transform (the reviewer's reference implementation), replaying the OLD exploratory
protocol unchanged (PCA-64 fitted on the whole source pool, random 5-fold in-port CV) so the effect of
the math error is isolated. Results here are explicitly NOT strict generalization estimates.

Columns per (fold, source port P):
  cross        logistic probe trained on the other source ports -> evaluated on P   (old protocol)
  coral_wrong  the previous (incorrect) transform: whiten(Cs) @ whiten(Ct)         (erratum reference)
  coral_t2s    corrected target->source: A=(Ct+rI)^-1/2 (Cs+rI)^1/2, source classifier frozen
  coral_s2t    corrected source->target: source mapped then a fresh classifier is fitted
  identity     transform applied to the same pool it was fitted from (should be ~identity)

usage: python e02_coral_regression.py --fold Rotterdam
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
OUT = ROOT / 'e02_alignment'
REF = KS / 'e02_alignment_protocol_v1'
MIN_PORT_N = 150
RIDGE = 1e-3          # CORAL whitening/re-colouring ridge, recorded in the output metadata
PCA_DIMS = 64
sys.path.insert(0, str(REF))
from coral_reference import fit_target_to_source, fit_source_to_target, symmetric_power  # noqa: E402


def sym_inv_sqrt(c, ridge):
    return symmetric_power(c + ridge * np.eye(c.shape[0]), -0.5)


def ba(y, pred, n_classes):
    rs = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--manifest', type=Path, default=KS / 'e01_first_batch' / 'split_manifest.json')
    a = ap.parse_args()
    tag = a.fold.replace(' ', '')
    d = np.load(SRCDEV / (tag + '_source.npz'), allow_pickle=False)
    z_raw, y, port = d['z'].astype(np.float64), d['y'], d['port'].astype(str)
    vocab = [str(x) for x in d['class_names']]
    C = len(vocab)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pca = PCA(n_components=min(PCA_DIMS, z_raw.shape[1]), random_state=20260920).fit(z_raw)  # OLD protocol
    z = pca.transform(z_raw)

    rows, meta = [], {'fold': a.fold, 'ridge': RIDGE, 'pca_dims': z.shape[1],
                      'pca_fit_scope': 'whole source pool (OLD exploratory protocol)',
                      'mapping_direction': 'target->source (source classifier frozen)',
                      'protocol_note': 'old protocol: PCA and random CV are NOT strict generalization',
                      'ports': []}
    for P in sorted(set(port.tolist())):
        m_p, m_o = port == P, port != P
        if m_p.sum() < MIN_PORT_N or len(set(y[m_p].tolist())) < 2 or len(set(y[m_o].tolist())) < 2:
            continue
        Xo, yo, Xp, yp = z[m_o], y[m_o], z[m_p], y[m_p]
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xo, yo)
        cross = clf.predict(Xp)

        # previous (incorrect) transform: Cs^-1/2 @ Ct^-1/2 with no re-colouring
        Cs = np.atleast_2d(np.cov(Xo, rowvar=False, ddof=1))
        Ct = np.atleast_2d(np.cov(Xp, rowvar=False, ddof=1))
        W_wrong = sym_inv_sqrt(Cs, RIDGE) @ sym_inv_sqrt(Ct, RIDGE)
        wrong = clf.predict((Xp - Xp.mean(0)) @ W_wrong + Xo.mean(0))
        res_wrong = float(np.linalg.norm(W_wrong.T @ (Cs + RIDGE * np.eye(Cs.shape[0])) @ W_wrong -
                                         (Ct + RIDGE * np.eye(Ct.shape[0])), ord='fro') /
                          max(np.linalg.norm(Ct, ord='fro'), 1e-12))

        # corrected directions via the reviewer reference
        t2s = fit_target_to_source(Xo, Xp, ridge=RIDGE)
        pred_t2s = clf.predict(t2s.transform(Xp))
        s2t = fit_source_to_target(Xo, Xp, ridge=RIDGE)
        pred_s2t = LogisticRegression(max_iter=3000, C=1.0).fit(s2t.transform(Xo), yo).predict(Xp)
        ident = t2s.transform(Xo)                     # fitted on Xp; applied to Xo -> not identity
        t2s_self = fit_target_to_source(Xo, Xo, ridge=RIDGE)
        iferr = float(np.abs(t2s_self.transform(Xo) - Xo).max())

        rows.append({'fold': a.fold, 'port': P, 'n': int(m_p.sum()), 'n_other': int(m_o.sum()),
                     'classes': len(set(yp.tolist())),
                     'cross_acc': float((cross == yp).mean()), 'cross_ba': ba(yp, cross, C),
                     'coral_wrong_acc': float((wrong == yp).mean()), 'coral_wrong_ba': ba(yp, wrong, C),
                     'coral_t2s_acc': float((pred_t2s == yp).mean()), 'coral_t2s_ba': ba(yp, pred_t2s, C),
                     'coral_s2t_acc': float((pred_s2t == yp).mean()), 'coral_s2t_ba': ba(yp, pred_s2t, C),
                     'cov_residual_t2s': t2s.regularized_covariance_residual(),
                     'cov_residual_wrong': res_wrong,
                     'identity_max_abs_err': iferr})
        meta['ports'].append({'port': P, 'n_target_adapt': int(m_p.sum()), 'n_source_reference': int(m_o.sum()),
                              'ridge': RIDGE, 'classes': len(set(yp.tolist()))})

    OUT.mkdir(parents=True, exist_ok=True)
    csvp = OUT / ('coral_regression_%s.csv' % tag)
    with csvp.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / ('coral_regression_%s_meta.json' % tag)).write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    for r in rows:
        print('%-10s %-20s cross %.3f | wrong %.3f | T2S %.3f | S2T %.3f | 残差 %.2e/%.2e | 恒等 %.1e' % (
            a.fold, r['port'][:20], r['cross_acc'], r['coral_wrong_acc'], r['coral_t2s_acc'],
            r['coral_s2t_acc'], r['cov_residual_t2s'], r['cov_residual_wrong'], r['identity_max_abs_err']))
    if rows:
        print('%-10s 平均 cross %.3f | wrong %.3f | T2S %.3f | S2T %.3f' % (
            a.fold, np.mean([r['cross_acc'] for r in rows]), np.mean([r['coral_wrong_acc'] for r in rows]),
            np.mean([r['coral_t2s_acc'] for r in rows]), np.mean([r['coral_s2t_acc'] for r in rows])))


if __name__ == '__main__':
    main()

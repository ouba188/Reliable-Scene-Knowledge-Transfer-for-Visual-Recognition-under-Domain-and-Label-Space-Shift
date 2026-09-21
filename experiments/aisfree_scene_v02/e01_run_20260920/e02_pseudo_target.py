"""E02 step 2: source-pseudo-target protocol + the three-column adaptation baseline.

Pseudo-target port q = a port in the fold's source_meta_query_ports (never part of that encoder's
supervised source fit). Its products are split by ACQUISITION DATE into
  U_q^adapt : unlabelled adaptation pool (earlier dates)   -- visual features only, no labels touched
  E_q^eval  : independent later-date products, labels used ONLY by the evaluator

Preprocessing/classifier rules (identical for every method):
  PCA-64 fitted on D_fit only; LogisticRegression(C=1, max_iter=3000) fitted on D_fit only.
Methods:
  cross        no adaptation
  mean_only    E_q features shifted by (mean(D_fit) - mean(U_q))     (unlabelled U_q only)
  coral_t2s    corrected target->source transport from U_q to D_fit  (unlabelled U_q only)
  coral_s2t    corrected source->target: D_fit mapped, fresh classifier
  in_ref       supervised reference: classifier trained on U_q (labels), evaluated on E_q
               (reference only, not an upper bound, label path isolated from all unsupervised methods)

usage: python e02_pseudo_target.py --fold Rotterdam --run /root/autodl-tmp/e01_runs/Rotterdam_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
REF = KS / 'e02_alignment_protocol_v1'
OUT = ROOT / 'e02_alignment'
RIDGE = 1e-3
PCA_DIMS = 64
MIN_EVAL, MIN_ADAPT, MIN_FIT = 30, 60, 120
sys.path.insert(0, str(REF))
from coral_reference import fit_target_to_source, fit_source_to_target  # noqa: E402


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def pdate(pid):
    parts = pid.split('_')
    for p in parts:
        if len(p) >= 15 and p[:8].isdigit() and p[8] == 'T':
            return p[:8]
    return ''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=KS / 'e01_first_batch' / 'split_manifest.json')
    a = ap.parse_args()
    tag = a.fold.replace(' ', '')
    fold = json.loads(a.manifest.read_text())['folds'][a.fold]
    fit_products = set(fold.get('source_fit_products', []))
    meta_products = set(fold.get('source_meta_query_products', []))
    meta_ports = set(fold.get('source_meta_query_ports', []))
    cal_products = set(fold.get('source_calibration_products', [])) | set(
        fold.get('source_calibration_adapt_products', []))
    cal_ports = set(fold.get('source_calibration_ports', []))
    vocab = [str(x) for x in json.loads((a.run / 'heads' / 'class_vocab.json').read_text())]
    C = len(vocab)
    vindex = {c: i for i, c in enumerate(vocab)}

    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    z = feats['z']; prod = feats['product_id'].astype(str); port = feats['port'].astype(str)
    ids = feats['sample_id'].astype(str)
    labels = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            oid = r['object_id']
            if r['deployable_offshore'] == '1':
                labels[oid] = (r.get('ais_final_class') or '').strip()

    def labelled_mask(sel):
        y = np.array([vindex.get(labels.get(ids[i], ''), -1) for i in np.flatnonzero(sel)])
        return y >= 0

    fit_sel = np.isin(prod, list(fit_products))
    y_fit_all = np.array([vindex.get(labels.get(o, ''), -1) for o in ids[fit_sel]])
    fit_lab = y_fit_all >= 0
    if fit_lab.sum() < MIN_FIT:
        print('%-10s D_fit 标注不足 (%d)' % (a.fold, int(fit_lab.sum()))); return
    X_fit = np.asarray(z[fit_sel][fit_lab], dtype=np.float64)
    y_fit = y_fit_all[fit_lab]

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    pca = PCA(n_components=min(PCA_DIMS, X_fit.shape[1], max(2, X_fit.shape[0] - 1)),
              random_state=20260920).fit(X_fit)          # fitted on D_fit ONLY
    Xf = pca.transform(X_fit)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xf, y_fit)

    rows, split_log = [], {'fold': a.fold, 'ridge': RIDGE, 'pca_dims': int(Xf.shape[1]),
                           'pca_fit_scope': 'source_fit only', 'classifier': 'LogisticRegression(C=1, max_iter=3000)',
                           'label_policy': 'labels read only by the evaluator; U_q^adapt is unlabelled',
                           'pseudo_targets': []}
    for q in sorted(meta_ports | cal_ports):
        # port membership comes from the feature table's own port column
        allowed = meta_products | cal_products
        qp = sorted({p for p, pt in zip(prod.tolist(), port.tolist()) if pt == q and p in allowed})
        q_sel = np.isin(prod, qp)
        if q_sel.sum() < MIN_ADAPT:
            continue
        dates = sorted({pdate(p) for p in qp if pdate(p)})
        if len(dates) < 2:
            split_log['pseudo_targets'].append({'port': q, 'assessable': False,
                                                'reason': 'single acquisition date', 'products': len(qp)})
            continue
        adapt_dates = set(dates[:max(1, len(dates) // 2)])
        p_adapt = {p for p in qp if pdate(p) in adapt_dates}
        p_eval = {p for p in qp if pdate(p) and pdate(p) not in adapt_dates}
        if not p_eval:
            continue
        a_sel, e_sel = np.isin(prod, list(p_adapt)), np.isin(prod, list(p_eval))
        Xa = np.asarray(z[a_sel], dtype=np.float64)
        Xe = np.asarray(z[e_sel], dtype=np.float64)
        y_e_all = np.array([vindex.get(labels.get(o, ''), -1) for o in ids[e_sel]])
        y_a_all = np.array([vindex.get(labels.get(o, ''), -1) for o in ids[a_sel]])
        keep_e = y_e_all >= 0
        if Xa.shape[0] < MIN_ADAPT or keep_e.sum() < MIN_EVAL:
            split_log['pseudo_targets'].append(
                {'port': q, 'assessable': False, 'reason': 'insufficient unlabelled adapt or labelled eval',
                 'n_adapt': int(Xa.shape[0]), 'n_eval_labeled': int(keep_e.sum()),
                 'adapt_products': len(p_adapt), 'eval_products': len(p_eval)})
            continue

        Xa_p, Xe_p = pca.transform(Xa), pca.transform(Xe)          # frozen preprocessing
        y_e = y_e_all[keep_e]
        Xe_k = Xe_p[keep_e]
        preds = {'cross': clf.predict(Xe_k)}
        preds['mean_only'] = clf.predict(Xe_k - Xa_p.mean(0) + Xf.mean(0))
        t2s = fit_target_to_source(Xf, Xa_p, ridge=RIDGE)
        preds['coral_t2s'] = clf.predict(t2s.transform(Xe_k))
        s2t = fit_source_to_target(Xf, Xa_p, ridge=RIDGE)
        preds['coral_s2t'] = LogisticRegression(max_iter=3000, C=1.0).fit(s2t.transform(Xf), y_fit).predict(Xe_k)
        keep_a = y_a_all >= 0
        if keep_a.sum() >= 30 and len(set(y_a_all[keep_a].tolist())) >= 2:
            preds['in_ref'] = LogisticRegression(max_iter=3000, C=1.0).fit(
                Xa_p[keep_a], y_a_all[keep_a]).predict(Xe_k)
        row = {'fold': a.fold, 'pseudo_target_port': q,
               'pseudo_target_role': 'meta' if q in meta_ports else 'cal', 'n_fit': int(Xf.shape[0]), 'n_adapt': int(Xa.shape[0]),
               'n_eval': int(keep_e.sum()), 'adapt_products': len(p_adapt), 'eval_products': len(p_eval),
               'adapt_dates': len(adapt_dates), 'eval_dates': len(dates) - len(adapt_dates),
               'cov_residual_t2s': t2s.regularized_covariance_residual(), 'ridge': RIDGE,
               'classes_in_eval': '|'.join(sorted({vocab[i] for i in set(y_e.tolist())}))}
        for k, v in preds.items():
            row['%s_acc' % k] = float((v == y_e).mean())
            row['%s_ba' % k] = ba(y_e, v, C)
        rows.append(row)
        split_log['pseudo_targets'].append(
            {'port': q, 'assessable': True, 'n_fit': int(Xf.shape[0]), 'n_adapt': int(Xa.shape[0]),
             'role': 'meta' if q in meta_ports else 'cal',
             'n_eval_labeled': int(keep_e.sum()), 'adapt_products': sorted(p_adapt),
             'eval_products': sorted(p_eval), 'adapt_dates': sorted(adapt_dates),
             'eval_dates': sorted(set(dates) - adapt_dates)})

    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = []
        for r in rows:                      # union of keys: in_ref only exists when the adapt pool has labels
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with (OUT / 'cross_mean_coral_corrected.csv').open('a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        for r in rows:
            print('%-10s q=%-12s fit %4d adapt %4d eval %4d | cross %.3f/%.3f  mean %.3f/%.3f  T2S %.3f/%.3f  S2T %.3f/%.3f%s' % (
                a.fold, r['pseudo_target_port'][:12], r['n_fit'], r['n_adapt'], r['n_eval'],
                r['cross_acc'], r['cross_ba'], r['mean_only_acc'], r['mean_only_ba'],
                r['coral_t2s_acc'], r['coral_t2s_ba'], r['coral_s2t_acc'], r['coral_s2t_ba'],
                ('  in_ref %.3f/%.3f' % (r['in_ref_acc'], r['in_ref_ba'])) if 'in_ref_acc' in r else ''))
    with (OUT / ('source_pseudo_target_splits_%s.json' % tag)).open('w', encoding='utf-8') as fh:
        json.dump(split_log, fh, ensure_ascii=False, indent=1)
    print('%-10s 可评估伪目标 %d / %d' % (a.fold, len(rows), len(split_log['pseudo_targets'])))


if __name__ == '__main__':
    main()

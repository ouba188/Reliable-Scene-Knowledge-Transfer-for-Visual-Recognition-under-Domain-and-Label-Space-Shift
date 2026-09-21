"""E02 single intervention: pair-aware observation-condition correction, validated leave-one-source-port-out.

Deployment-legal inputs only: the labelled source fit set D_fit and the UNLABELLED statistics of the
adaptation pool. The decision signal is a normalised mean-shift distance between D_fit and U_q in
source-fit PCA coordinates:

    s = || mean(D_fit) - mean(U_q) || / sqrt(mean(diag(Cov(D_fit))))

Rule: apply the mean-only correction when s >= threshold, else stay with `cross`.
The threshold is fitted on SOURCE ports only (leave-one-source-port-out among the fold's source fit
ports: train on the other ports, treat the held-out port as an unlabelled adaptation pool), then applied
unchanged to each source-pseudo-target pair. Labels of the held-out port are read only by the evaluator.

usage: python e02_pair_rule.py --fold Rotterdam --run /root/autodl-tmp/e01_runs/Rotterdam_rev1
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
OUT = ROOT / 'e02_alignment'
MIN_PORT_LAB, MIN_PORT_ALL = 60, 200


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=KS / 'e01_first_batch' / 'split_manifest.json')
    a = ap.parse_args()
    tag = a.fold.replace(' ', '')
    fold = json.loads(a.manifest.read_text())['folds'][a.fold]
    vocab = [str(x) for x in json.loads((a.run / 'heads' / 'class_vocab.json').read_text())]
    C = len(vocab)
    vindex = {c: i for i, c in enumerate(vocab)}

    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    z, port, ids = feats['z'], feats['port'].astype(str), feats['sample_id'].astype(str)
    labels = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                labels[r['object_id']] = (r.get('ais_final_class') or '').strip()
    y_all = np.array([vindex.get(labels.get(o, ''), -1) for o in ids])

    p2port = {}
    for p, pt in zip(feats['product_id'].astype(str).tolist(), port.tolist()):
        p2port[p] = pt
    fit_ports = sorted({p2port[p] for p in fold.get('source_fit_products', []) if p in p2port})
    fit_ports = [q for q in fit_ports if (port == q).sum() >= MIN_PORT_ALL]

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    def probe(train_sel, test_sel):
        tr_lab = train_sel & (y_all >= 0)
        if tr_lab.sum() < 60 or len(set(y_all[tr_lab].tolist())) < 2:
            return None
        Xtr = np.asarray(z[tr_lab], dtype=np.float64)
        pca = PCA(n_components=min(64, Xtr.shape[1], max(2, Xtr.shape[0] - 1)),
                  random_state=20260920).fit(Xtr)          # fitted on the training source only
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(pca.transform(Xtr), y_all[tr_lab])
        Xtr_p = pca.transform(Xtr)
        ref_mu = pca.transform(np.asarray(z[train_sel], dtype=np.float64).mean(0)[None])[0]
        ref_scale = float(np.sqrt(np.mean(np.var(Xtr_p, axis=0))) + 1e-9)
        return clf, pca, ref_mu, ref_scale

    def signal(pca, ref_mu, ref_scale, X_adapt):
        """legal signal: unlabelled mean-shift in PCA coordinates, normalised by the source scale."""
        mu_a = pca.transform(np.asarray(X_adapt, dtype=np.float64).mean(0)[None])[0]
        return float(np.linalg.norm(ref_mu - mu_a) / ref_scale)

    # ---------- source-side development (leave-one-source-port-out) ----------
    dev = []
    for P in fit_ports:
        other = np.isin(port, [q for q in fit_ports if q != P])
        held = port == P
        got = probe(other & (y_all >= 0), held)
        if got is None:
            continue
        clf, pca, ref_mu, ref_scale = got
        Xa = np.asarray(z[held], dtype=np.float64)
        Xa_p = pca.transform(Xa)
        sel_e = held & (y_all >= 0)
        if sel_e.sum() < MIN_PORT_LAB:
            continue
        Xe_p = pca.transform(np.asarray(z[sel_e], dtype=np.float64))
        y_e = y_all[sel_e]
        cross = clf.predict(Xe_p)
        meanp = clf.predict(Xe_p - Xa_p.mean(0) + ref_mu)
        dev.append({'port': P, 'n_eval': int(sel_e.sum()), 'signal': signal(pca, ref_mu, ref_scale, Xa),
                    'cross_acc': float((cross == y_e).mean()), 'cross_ba': ba(y_e, cross, C),
                    'mean_acc': float((meanp == y_e).mean()), 'mean_ba': ba(y_e, meanp, C)})
    if not dev:
        print('%-10s 源端留出不足，无法拟合规则' % a.fold); return

    # threshold search maximising total source-side gain (tie -> prefer the smaller threshold)
    cand = sorted({round(d['signal'], 6) for d in dev} | {-1.0})
    best_t, best_g = -1.0, -1e9
    for t in cand:
        g = sum((d['mean_acc'] - d['cross_acc']) if d['signal'] >= t else 0.0 for d in dev)
        if g > best_g + 1e-12:
            best_t, best_g = t, g
    dev_gain = {'cross_acc': float(np.mean([d['cross_acc'] for d in dev])),
                'mean_acc': float(np.mean([d['mean_acc'] for d in dev])),
                'rule_acc': float(np.mean([d['mean_acc'] if d['signal'] >= best_t else d['cross_acc'] for d in dev]))}

    # ---------- apply to the pseudo-target pairs ----------
    ref = OUT / 'cross_mean_coral_corrected.csv'
    rows = [r for r in csv.DictReader(ref.open(encoding='utf-8')) if r['fold'] == a.fold]
    res = {'fold': a.fold, 'threshold': best_t, 'source_development': dev,
           'source_development_gain': dev_gain, 'applied': []}
    pmap = np.array([p2port.get(p, '') for p in feats['product_id'].astype(str).tolist()])
    fit_sel = np.isin(pmap, fit_ports)
    got_fit = probe(fit_sel & (y_all >= 0), None)
    if got_fit is None:
        (OUT / ('pair_rule_%s.json' % tag)).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print('%-10s 无法在 D_fit 上重建分类器' % a.fold); return
    clf_f, pca_f, ref_mu_f, ref_scale_f = got_fit
    for r in rows:
        q = r['pseudo_target_port']
        q_sel = port == q
        if q_sel.sum() < 20:
            continue
        U = np.asarray(z[q_sel], dtype=np.float64)
        s = signal(pca_f, ref_mu_f, ref_scale_f, U)
        use_mean = s >= best_t
        r2 = dict(r)
        r2['signal'] = s
        r2['rule_choice'] = 'mean_only' if use_mean else 'cross'
        r2['rule_acc'] = r2['mean_only_acc'] if use_mean else r2['cross_acc']
        r2['rule_ba'] = r2['mean_only_ba'] if use_mean else r2['cross_ba']
        r2['best_of_cross_mean_acc'] = max(float(r['cross_acc']), float(r['mean_only_acc']))
        r2['oracle_acc'] = max(float(r['cross_acc']), float(r['mean_only_acc']), float(r['coral_t2s_acc']))
        res['applied'].append(r2)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('pair_rule_%s.json' % tag)).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print('%-10s 阈值 %.4f（源端：cross %.3f → mean %.3f → RULE %.3f）' % (
        a.fold, best_t, dev_gain['cross_acc'], dev_gain['mean_acc'], dev_gain['rule_acc']))
    for r in res['applied']:
        if int(r['n_eval']) < 200:
            continue
        print('   q=%-10s n=%5s signal %.3f → %-9s | cross %.3f  mean %.3f  T2S %.3f  RULE %.3f  (oracle %.3f)' % (
            r['pseudo_target_port'][:10], r['n_eval'], r['signal'], r['rule_choice'],
            float(r['cross_acc']), float(r['mean_only_acc']), float(r['coral_t2s_acc']),
            float(r['rule_acc']), r['oracle_acc']))


if __name__ == '__main__':
    main()

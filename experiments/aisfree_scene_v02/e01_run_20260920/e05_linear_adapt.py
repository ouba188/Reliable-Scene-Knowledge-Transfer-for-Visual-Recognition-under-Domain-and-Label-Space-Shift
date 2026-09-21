"""E05: source-supervised linear representation adaptation, evaluated on the frozen E02 protocol.

Inputs allowed: D_fit features+labels, U_q unlabelled features, E_q features (labels only in the evaluator).
Module: linear map W (64x64, or diagonal) applied to both sides, trained to minimise
    CE(W z_fit, y_fit) + lam * ||Cov(W z_fit) - Cov(z_adapt)||_F^2 + mu * ||mean(W z_fit) - mean(z_adapt)||^2
W starts at identity, so (lam=mu=0) reproduces Cross. The classifier is refitted on W z_fit with the
frozen settings LogisticRegression(C=1, max_iter=3000) and predicts W z_eval.

Hyper-parameters are selected ONLY on source-side held-out ports (leave-one-port-out episodes), then
applied unchanged to the five pseudo-target pairs. Baselines recomputed in the same run for exact
comparability.

usage: python e05_linear_adapt.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
E02 = ROOT / 'e02_alignment'
OUT = ROOT / 'e05_linear_adapt'
PCA_DIMS = 64
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'Fujairah', 'JebelAli', 'PortSaid']
META = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortKlang': 'Port Klang',
        'Fujairah': 'Fujairah', 'JebelAli': 'Jebel Ali', 'PortSaid': 'Port Said'}
GRID = list(product([0.0, 0.1, 1.0, 10.0], [0.0, 1.0]))      # (lam, mu)


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def train_map(Xf, y_fit, Xa, lam, mu, diag=False, steps=300, lr=1e-2, seed=20260920):
    import torch
    torch.manual_seed(seed)
    d = Xf.shape[1]
    C = int(y_fit.max()) + 1
    Xt = torch.tensor(Xf, dtype=torch.float64)
    yt = torch.tensor(y_fit, dtype=torch.long)
    Xat = torch.tensor(Xa, dtype=torch.float64)
    if diag:
        s = torch.ones(d, dtype=torch.float64, requires_grad=True)
        params, apply = [s], (lambda Z: Z * s)
    else:
        W = torch.eye(d, dtype=torch.float64, requires_grad=True)
        params, apply = [W], (lambda Z: Z @ W.T)
    clf = torch.nn.Linear(d, C).double()
    opt = torch.optim.Adam(list(clf.parameters()) + params, lr=lr)
    tgt_mu = Xat.mean(0)
    tgt_cov = torch.cov(Xat.T)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        Z = apply(Xt)
        loss = lossf(clf(Z), yt)
        Za = apply(Xat)
        if lam > 0:
            loss = loss + lam * torch.linalg.matrix_norm(torch.cov(Za.T) - tgt_cov) ** 2
        if mu > 0:
            loss = loss + mu * torch.linalg.vector_norm(Za.mean(0) - tgt_mu) ** 2
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (lambda Z: Z * s.detach().numpy()) if diag else (lambda Z: Z @ W.detach().numpy().T)


def source_episodes():
    """leave-one-port-out episodes from the source caches (fit-labelled reference, held-out port as adapt)."""
    man = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']
    eps = []
    for tag in FOLDS:
        d = np.load(SRCDEV / (tag + '_source.npz'), allow_pickle=False)
        z, y, port = d['z'].astype(np.float64), d['y'], d['port'].astype(str)
        fit_ports = set(man[META[tag]].get('source_fit_ports', []))
        for P in sorted(set(port.tolist())):
            m_ref = np.array([(pt != P) and (pt in fit_ports) for pt in port.tolist()])
            m_a = port == P
            if m_ref.sum() < 200 or m_a.sum() < 150 or len(set(y[m_ref].tolist())) < 2 \
               or len(set(y[m_a].tolist())) < 2:
                continue
            eps.append((tag, P, z[m_ref], y[m_ref], z[m_a], y[m_a]))
    return eps


def load_labels(products):
    lab = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1' and r['product_id'] in products:
                lab[r['object_id']] = (r.get('ais_final_class') or '').strip()
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- hyper-parameter selection on source-side held-out ports only ----
    eps = source_episodes()
    print('源端 episode %d 个；网格 %d 组' % (len(eps), len(GRID)))
    scores = {g: [] for g in GRID}
    for tag, P, zr, yr, za, ya in eps:
        pca = PCA(n_components=min(PCA_DIMS, zr.shape[1], max(2, zr.shape[0] - 1)),
                  random_state=20260920).fit(zr)
        Xr, Xa = pca.transform(zr), pca.transform(za)
        C = int(max(yr.max(), ya.max())) + 1
        for g in GRID:
            W = train_map(Xr, yr, Xa, g[0], g[1])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(W(Xr), yr)
            scores[g].append(ba(ya, clf.predict(W(Xa)), C))
    mean_scores = {g: float(np.mean(v)) for g, v in scores.items()}
    best = max(mean_scores.items(), key=lambda kv: kv[1])
    print('源端选择：lam=%.2f mu=%.2f（源端平均 BA %.3f；λ=μ=0 时 %.3f）' % (
        best[0][0], best[0][1], best[1], mean_scores[(0.0, 0.0)]))

    # ---- apply to the five pseudo-target pairs (frozen E02 splits) ----
    rows = []
    for tag in FOLDS:
        sp = json.loads((E02 / ('source_pseudo_target_splits_%s.json' % tag)).read_text())
        pairs = [p for p in sp['pseudo_targets'] if p.get('assessable') and p['n_eval_labeled'] >= 400]
        if not pairs:
            continue
        feats = np.load(ROOT / 'e01_runs' / (META[tag] + '_rev1') / 'features.npz', allow_pickle=False)
        z = feats['z']; prod = feats['product_id'].astype(str); ids = feats['sample_id'].astype(str)
        vocab = json.loads((ROOT / 'e01_runs' / (META[tag] + '_rev1') / 'heads' / 'class_vocab.json').read_text())
        vindex = {c: i for i, c in enumerate(vocab)}
        C = len(vocab)
        fitp = set(json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())
                   ['folds'][META[tag]].get('source_fit_products', []))
        lab_fit = load_labels(fitp)                      # one table scan per fold
        sel_f = np.array([i for i, o in enumerate(ids) if o in lab_fit and lab_fit[o] in vindex])
        for pr in pairs:
            lab_all = load_labels(set(pr['adapt_products']) | set(pr['eval_products']))
            sel_a = np.array([i for i, p in enumerate(prod.tolist()) if p in set(pr['adapt_products'])])
            sel_e = np.array([i for i, o in enumerate(ids) if o in lab_all and lab_all[o] in vindex
                              and prod[i] in set(pr['eval_products'])])
            if len(sel_f) < 200 or len(sel_a) < 100 or len(sel_e) < 100:
                continue
            pca = PCA(n_components=min(PCA_DIMS, z.shape[1], len(sel_f) - 1), random_state=20260920).fit(z[sel_f])
            Xf, yf = pca.transform(z[sel_f]).astype(np.float64), np.array([vindex[lab_fit[ids[i]]] for i in sel_f])
            Xa = pca.transform(z[sel_a]).astype(np.float64)
            Xe, ye = pca.transform(z[sel_e]).astype(np.float64), np.array([vindex[lab_all[ids[i]]] for i in sel_e])
            clf0 = LogisticRegression(max_iter=3000, C=1.0).fit(Xf, yf)
            rec = {'fold': tag, 'pseudo_target': pr['port'], 'n_fit': len(sel_f), 'n_adapt': len(sel_a),
                   'n_eval': len(sel_e), 'lam': best[0][0], 'mu': best[0][1],
                   'cross_acc': float((clf0.predict(Xe) == ye).mean()), 'cross_ba': ba(ye, clf0.predict(Xe), C),
                   'mean_acc': float((clf0.predict(Xe - Xa.mean(0) + Xf.mean(0)) == ye).mean()),
                   'mean_ba': ba(ye, clf0.predict(Xe - Xa.mean(0) + Xf.mean(0)), C)}
            W = train_map(Xf, yf, Xa, best[0][0], best[0][1])
            clfm = LogisticRegression(max_iter=3000, C=1.0).fit(W(Xf), yf)
            pm = clfm.predict(W(Xe))
            rec['module_acc'], rec['module_ba'] = float((pm == ye).mean()), ba(ye, pm, C)
            Wd = train_map(Xf, yf, Xa, best[0][0], best[0][1], diag=True)
            clfd = LogisticRegression(max_iter=3000, C=1.0).fit(Wd(Xf), yf)
            pd_ = clfd.predict(Wd(Xe))
            rec['diag_acc'], rec['diag_ba'] = float((pd_ == ye).mean()), ba(ye, pd_, C)
            rec['best_baseline_acc'] = max(rec['cross_acc'], rec['mean_acc'])
            rec['best_baseline_ba'] = max(rec['cross_ba'], rec['mean_ba'])
            rows.append(rec)
            print('%-10s q=%-10s n=%4d | cross %.3f/%.3f  mean %.3f/%.3f | module %.3f/%.3f  diag %.3f/%.3f' % (
                tag, pr['port'], rec['n_eval'], rec['cross_acc'], rec['cross_ba'], rec['mean_acc'],
                rec['mean_ba'], rec['module_acc'], rec['module_ba'], rec['diag_acc'], rec['diag_ba']))

    with (OUT / 'linear_adapt_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {'grid': GRID, 'source_mean_ba': {('%g' % g[0] + ',' + '%g' % g[1]): v for g, v in mean_scores.items()},
               'selected': {'lam': best[0][0], 'mu': best[0][1], 'source_mean_ba': best[1]},
               'baseline_lam0_mu0_ba': mean_scores[(0.0, 0.0)],
               'module_vs_best_baseline_ba': float(np.mean([r['module_ba'] - r['best_baseline_ba'] for r in rows])),
               'module_vs_cross_ba': float(np.mean([r['module_ba'] - r['cross_ba'] for r in rows])),
               'diag_vs_best_baseline_ba': float(np.mean([r['diag_ba'] - r['best_baseline_ba'] for r in rows]))}
    (OUT / 'linear_adapt_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print('\n模块 vs 最好基线（BA）：%+.3f ；vs Cross：%+.3f ；对角版 vs 最好基线：%+.3f' % (
        summary['module_vs_best_baseline_ba'], summary['module_vs_cross_ba'], summary['diag_vs_best_baseline_ba']))


if __name__ == '__main__':
    main()

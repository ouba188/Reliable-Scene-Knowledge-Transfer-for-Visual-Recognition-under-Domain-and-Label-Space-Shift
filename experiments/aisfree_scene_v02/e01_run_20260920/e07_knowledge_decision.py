"""E07: knowledge as a DECISION signal under the corrected protocol (single intervention).

Idea: the relation models predict per-class moment profiles (phi[j, x, c, d]); the object carries an
observed moment profile m[x, d] (image-derived, deployment-legal). Reweight the frozen classifier:
    score(c|x) = p(c|x) * exp(-lam * d(x, c)),  d(x,c) = mean_j sum_{d enabled} ((phi - m)/sigma_d)^2
lambda is selected ONLY on source-port held-out episodes that exclude every pseudo-target port used in
the evaluation (fixing the E05 selection-path criticism). Two controls:
  * lam = 0            -> exactly the Cross baseline
  * shuffled knowledge -> same term with the class<->moment association permuted (proves the signal
                          comes from the knowledge, not from adding a term)

usage: python e07_knowledge_decision.py
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
IN = KS / 'e01_inputs'
E02 = ROOT / 'e02_alignment'
OUT = ROOT / 'e07_knowledge_decision'
SRCDEV = ROOT / 'e01_runs' / 'aux_source_dev'
PCA_DIMS = 64
LAMBDAS = [0.0, 0.05, 0.2, 1.0, 5.0]
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'Fujairah', 'JebelAli', 'PortSaid']
META = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortKlang': 'Port Klang',
        'Fujairah': 'Fujairah', 'JebelAli': 'Jebel Ali', 'PortSaid': 'Port Said'}
sys.path.insert(0, str(IN))


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def knowledge_term(phi, m, enabled, sigma):
    """phi[J,N,C,D], m[N,D] -> d[N,C]; enabled: bool[D]; sigma: [D] normaliser."""
    J, N, C, D = phi.shape
    d = np.zeros((N, C), dtype=np.float64)
    for j in range(J):
        diff = (phi[j] - m[:, None, :]) / sigma[None, None, :]
        d += (diff[..., enabled] ** 2).sum(-1)
    return d / max(1, J)


def main() -> None:
    import e01_glue_rev1 as glue
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']

    # ---------- 1) lambda selection on source episodes that EXCLUDE every pseudo-target port ----------
    target_ports = set()
    pairs_by_fold = {}
    for tag in FOLDS:
        sp = json.loads((E02 / ('source_pseudo_target_splits_%s.json' % tag)).read_text())
        pairs = [p for p in sp['pseudo_targets'] if p.get('assessable') and p['n_eval_labeled'] >= 400]
        pairs_by_fold[tag] = pairs
        target_ports |= {p['port'] for p in pairs}
    print('评估用伪目标港（选择时必须排除）:', sorted(target_ports), flush=True)

    sel = {lam: [] for lam in LAMBDAS}
    for tag in FOLDS:
        run = ROOT / 'e01_runs' / (META[tag] + '_rev1')
        d = np.load(SRCDEV / (tag + '_source.npz'), allow_pickle=False)
        z_raw, y, port = d['z'].astype(np.float64), d['y'], d['port'].astype(str)
        m_all = d['m'].astype(np.float64) if 'm' in d.files else None
        if m_all is None:
            continue
        feats = np.load(run / 'features.npz', allow_pickle=False)
        ids = feats['sample_id'].astype(str)
        pos = {s: i for i, s in enumerate(ids)}
        idx_all = np.array([pos.get(s, -1) for s in d['sample_id'].astype(str)], dtype=np.int64)
        ok = idx_all >= 0
        if ok.sum() < 200:
            continue
        sc = np.load(run / 'heads' / 'feature_scaler.npz')
        phi, _ = glue.rel_means(run, feats, idx_all[ok], [str(x) for x in d['class_names']], sc)
        rman = json.loads((run / 'relations' / 'relation_manifest.json').read_text())
        enabled = np.array(rman['enabled_dimensions'], dtype=bool)
        sigma = np.std(m_all[ok], axis=0)
        sigma[sigma < 1e-6] = 1.0
        y_ok, port_ok = y[ok], port[ok]
        fit_ports = set(man[META[tag]].get('source_fit_ports', []))
        for P in sorted(set(port_ok.tolist())):
            if P in target_ports or P not in fit_ports:
                continue                                    # exclude pseudo-target ports and non-fit roles
            mref = np.array([(pt != P) for pt in port_ok.tolist()])
            mheld = port_ok == P
            if mref.sum() < 200 or mheld.sum() < 150 or len(set(y_ok[mheld].tolist())) < 2:
                continue
            pca = PCA(n_components=min(PCA_DIMS, z_raw.shape[1], mref.sum() - 1), random_state=20260920).fit(
                z_raw[ok][mref])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(pca.transform(z_raw[ok][mref]), y_ok[mref])
            Pheld = clf.predict_proba(pca.transform(z_raw[ok][mheld]))
            dt = knowledge_term(phi[:, mheld], m_all[ok][mheld], enabled, sigma)
            C = Pheld.shape[1]
            for lam in LAMBDAS:
                logit = np.log(np.clip(Pheld, 1e-12, 1)) - lam * dt[:, :C]
                sel[lam].append(ba(y_ok[mheld], logit.argmax(1), C))
    mean_sel = {lam: float(np.mean(v)) for lam, v in sel.items() if v}
    best_lam = max(mean_sel.items(), key=lambda kv: kv[1]) if mean_sel else (0.0, float('nan'))
    print('源端选择（已排除伪目标港）: %s' % json.dumps({str(k): round(v, 4) for k, v in mean_sel.items()}),
          flush=True)
    print('选中 λ=%.2f（λ=0 即 Cross: %.4f）' % (best_lam[0], mean_sel.get(0.0, float('nan'))), flush=True)

    # ---------- 2) evaluate on the frozen pseudo-target pairs ----------
    rows = []
    for tag in FOLDS:
        pairs = pairs_by_fold[tag]
        if not pairs:
            continue
        run = ROOT / 'e01_runs' / (META[tag] + '_rev1')
        vocab = [str(x) for x in json.loads((run / 'heads' / 'class_vocab.json').read_text())]
        vindex = {c: i for i, c in enumerate(vocab)}
        C = len(vocab)
        feats = np.load(run / 'features.npz', allow_pickle=False)
        z = feats['z']; m = feats['m'].astype(np.float64)
        prod = feats['product_id'].astype(str); ids = feats['sample_id'].astype(str)
        fitp = set(man[META[tag]].get('source_fit_products', []))
        lab_fit = {}
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['deployable_offshore'] == '1' and r['product_id'] in fitp:
                    c = (r.get('ais_final_class') or '').strip()
                    if c in vindex:
                        lab_fit[r['object_id']] = c
        sel_f = np.array([i for i, o in enumerate(ids) if o in lab_fit])
        pca = PCA(n_components=min(PCA_DIMS, z.shape[1], len(sel_f) - 1), random_state=20260920).fit(z[sel_f])
        Xf = pca.transform(z[sel_f]); yf = np.array([vindex[lab_fit[ids[i]]] for i in sel_f])
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xf, yf)
        sc = np.load(run / 'heads' / 'feature_scaler.npz')
        sigma = np.std(m[sel_f], axis=0); sigma[sigma < 1e-6] = 1.0
        need_prods = set()
        for pr in pairs:
            need_prods |= set(pr['adapt_products']) | set(pr['eval_products'])
        need_idx = np.array([i for i, p in enumerate(prod.tolist()) if p in need_prods], dtype=np.int64)
        phi, _ = glue.rel_means(run, feats, need_idx, vocab, sc)           # [J, len(need_idx), C, D]
        phi_of = {int(i): k for k, i in enumerate(need_idx)}
        rman = json.loads((run / 'relations' / 'relation_manifest.json').read_text())
        enabled = np.array(rman['enabled_dimensions'], dtype=bool)
        for pr in pairs:
            keep_prods = set(pr['adapt_products']) | set(pr['eval_products'])
            lab_e = {}
            with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
                for r in csv.DictReader(fh):
                    if r['deployable_offshore'] == '1' and r['product_id'] in keep_prods:
                        c = (r.get('ais_final_class') or '').strip()
                        if c in vindex:
                            lab_e[r['object_id']] = c
            e_sel = np.array([i for i, o in enumerate(ids) if o in lab_e and prod[i] in set(pr['eval_products'])])
            if len(e_sel) < 100:
                continue
            ye = np.array([vindex[lab_e[ids[i]]] for i in e_sel])
            Pe = clf.predict_proba(pca.transform(z[e_sel]))
            dt = knowledge_term(phi[:, [phi_of[int(i)] for i in e_sel]], m[e_sel], enabled, sigma)
            base = np.log(np.clip(Pe, 1e-12, 1)).argmax(1)
            kn = (np.log(np.clip(Pe, 1e-12, 1)) - best_lam[0] * dt[:, :C]).argmax(1)
            rng = np.random.default_rng(20260920)
            perm = rng.permutation(C)
            sh = (np.log(np.clip(Pe, 1e-12, 1)) - best_lam[0] * dt[:, perm]).argmax(1)
            rows.append({'fold': tag, 'pseudo_target': pr['port'], 'lambda': best_lam[0],
                         'n_eval': len(e_sel), 'enabled_dims': int(enabled.sum()),
                         'cross_acc': float((base == ye).mean()), 'cross_ba': ba(ye, base, C),
                         'knowledge_acc': float((kn == ye).mean()), 'knowledge_ba': ba(ye, kn, C),
                         'shuffled_acc': float((sh == ye).mean()), 'shuffled_ba': ba(ye, sh, C),
                         'rescue': int(((base != ye) & (kn == ye)).sum()),
                         'harm': int(((base == ye) & (kn != ye)).sum()),
                         'class_support': json.dumps({vocab[c]: int((ye == c).sum()) for c in range(C)
                                                      if (ye == c).any()}, ensure_ascii=False)})
            print('%-10s q=%-10s n=%4d λ=%.2f | Cross %.3f/%.3f  知识 %.3f/%.3f  打乱 %.3f/%.3f  rescue %d harm %d' % (
                tag, pr['port'], len(e_sel), best_lam[0], rows[-1]['cross_acc'], rows[-1]['cross_ba'],
                rows[-1]['knowledge_acc'], rows[-1]['knowledge_ba'], rows[-1]['shuffled_acc'],
                rows[-1]['shuffled_ba'], rows[-1]['rescue'], rows[-1]['harm']), flush=True)
    if rows:
        with (OUT / 'knowledge_decision_results.csv').open('w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print('\n知识 − Cross（BA）：%+.3f ；打乱 − Cross（BA）：%+.3f' % (
            float(np.mean([r['knowledge_ba'] - r['cross_ba'] for r in rows])),
            float(np.mean([r['shuffled_ba'] - r['cross_ba'] for r in rows]))))
    (OUT / 'knowledge_decision_selection.json').write_text(json.dumps(
        {'lambda_grid_source_ba': mean_sel, 'lambda_selected': best_lam[0],
         'excluded_pseudo_target_ports': sorted(target_ports)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

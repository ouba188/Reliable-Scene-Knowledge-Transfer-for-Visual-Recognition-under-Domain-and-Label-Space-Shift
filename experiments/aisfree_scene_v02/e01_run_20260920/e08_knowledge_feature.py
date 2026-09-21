"""E08: port knowledge as per-object OBSERVATION features (relations.npz phi[N,4]), CPU-only.

Arms (same frozen encoder, PCA-64 fit on the training source only, LR C=1 max_iter=3000):
  A0  = [pca(z)]                              -> Cross baseline
  A1  = [pca(z), K_enabled]                   -> + per-object static-map knowledge features
  A2  = [pca(z), K_enabled(permuted)]         -> wrong-knowledge control (permute within same product
                                                   and same availability pattern; mask unchanged)
Support rule (preset, not tuned on target): a dim is enabled iff among source-fit labelled instances
  valid observations >= 30  AND  distinct source ports >= 2  AND  variance >= 1e-4.
Missing values are imputed with the training-source per-dim mean (no eval-side re-fitting).

Deliverables: feature_schema.json, train_and_scaler_lineage.json, source_feature_support.csv,
feature_arms_metrics.csv, per_class_and_per_pair_metrics.csv, DECISION.md (written separately).

usage: python e08_knowledge_feature.py
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
E02 = ROOT / 'e02_alignment'
REL = ROOT / 'e01_runs' / 'relations' / 'relations.npz'
OUT = ROOT / 'e08_knowledge_feature'
PCA_DIMS = 64
MIN_VALID, MIN_PORTS, MIN_VAR = 30, 2, 1e-4
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'Fujairah', 'JebelAli', 'PortSaid']
META = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortKlang': 'Port Klang',
        'Fujairah': 'Fujairah', 'JebelAli': 'Jebel Ali', 'PortSaid': 'Port Said'}


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']
    rel = np.load(REL, allow_pickle=False)
    rel_pos = {str(s): i for i, s in enumerate(rel['sample_id'])}
    rel_phi = rel['phi'].astype(np.float64)
    rel_avail = rel['available']
    rel_prod = rel['product_id'].astype(str)

    schema = {'source': str(REL), 'dims': {0: 'D0 density proxy', 1: 'D1 anchorage layer',
                                            2: 'D2 fairway layer', 3: 'D3 coastline context'},
              'imputation': 'training-source per-dim mean', 'available_mask': 'relations.npz available'}
    lineage, support_rows, arms_rows, perpair_rows = {}, [], [], []

    for tag in FOLDS:
        run = ROOT / 'e01_runs' / (META[tag] + '_rev1')
        vocab = [str(x) for x in json.loads((run / 'heads' / 'class_vocab.json').read_text())]
        vindex = {c: i for i, c in enumerate(vocab)}
        C = len(vocab)
        feats = np.load(run / 'features.npz', allow_pickle=False)
        z = feats['z']; ids = feats['sample_id'].astype(str)
        prod = feats['product_id'].astype(str); port = feats['port'].astype(str)
        fitp = set(man[META[tag]].get('source_fit_products', []))
        lab = {}
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['deployable_offshore'] == '1' and r['product_id'] in fitp:
                    c = (r.get('ais_final_class') or '').strip()
                    if c in vindex:
                        lab[r['object_id']] = c
        sel_f = np.array([i for i, o in enumerate(ids) if o in lab])
        rel_idx = np.array([rel_pos.get(o, -1) for o in ids[sel_f]], dtype=np.int64)
        ok_rel = rel_idx >= 0
        sel_f = sel_f[ok_rel]; rel_idx = rel_idx[ok_rel]
        K_f = rel_phi[rel_idx]                     # [n_fit, 4]
        A_f = rel_avail[rel_idx]
        pf = port[sel_f]
        y_f = np.array([vindex[lab[ids[i]]] for i in sel_f])

        enabled, support = [], {}
        for d in range(4):
            m = A_f[:, d] & np.isfinite(K_f[:, d])
            n_valid = int(m.sum())
            n_ports = len(set(pf[m].tolist()))
            var = float(K_f[m, d].var()) if n_valid > 1 else 0.0
            support[d] = {'n_valid': n_valid, 'n_ports': n_ports, 'variance': var,
                          'enabled': n_valid >= MIN_VALID and n_ports >= MIN_PORTS and var >= MIN_VAR}
            if support[d]['enabled']:
                enabled.append(d)
        for d in range(4):
            support[d].update({'dim': d, 'name': schema['dims'][d]})
            support_rows.append({'fold': tag, **support[d]})
        if not enabled:
            print('%-10s 无启用维（跳过）' % tag, flush=True)
            continue
        K_f = K_f[:, enabled]
        mean_f = np.nanmean(K_f, axis=0)          # imputation, fit on training source only
        std_f = np.nanstd(K_f, axis=0); std_f[std_f < 1e-9] = 1.0
        K_f = np.where(np.isfinite(K_f), K_f, mean_f)
        K_f = (K_f - mean_f) / std_f
        pca = PCA(n_components=min(PCA_DIMS, z.shape[1], len(sel_f) - 1), random_state=20260920).fit(z[sel_f])
        Zf = pca.transform(z[sel_f])
        A0 = LogisticRegression(max_iter=3000, C=1.0).fit(Zf, y_f)
        A1 = LogisticRegression(max_iter=3000, C=1.0).fit(np.concatenate([Zf, K_f], 1), y_f)
        rng = np.random.default_rng(20260920)
        Kf_sh = np.zeros_like(K_f)
        for p in set(prod[sel_f].tolist()):
            grp = prod[sel_f] == p
            Kf_sh[grp] = rng.permutation(K_f[grp])          # permute within same product (mask unchanged)
        A2 = LogisticRegression(max_iter=3000, C=1.0).fit(np.concatenate([Zf, Kf_sh], 1), y_f)
        lineage[tag] = {'encoder': META[tag] + '_rev1', 'pca_fit_on': 'source-fit labelled features only',
                        'knowledge_imputation_mean_std_fit_on': 'source-fit only',
                        'enabled_dims': enabled, 'n_fit': int(len(sel_f))}

        sp = json.loads((E02 / ('source_pseudo_target_splits_%s.json' % tag)).read_text())
        pairs = [p for p in sp['pseudo_targets'] if p.get('assessable') and p['n_eval_labeled'] >= 400]
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
            eri = np.array([rel_pos.get(o, -1) for o in ids[e_sel]], dtype=np.int64)
            ok_e = eri >= 0
            e_sel, eri, ye = e_sel[ok_e], eri[ok_e], ye[ok_e]
            Ze = pca.transform(z[e_sel])
            Ke = rel_phi[eri][:, enabled]
            Ke = np.where(np.isfinite(Ke), Ke, mean_f)
            Ke = (Ke - mean_f) / std_f
            rng2 = np.random.default_rng(20260920)
            Ke_sh = np.zeros_like(Ke)
            for p in set(prod[e_sel].tolist()):
                grp = prod[e_sel] == p
                Ke_sh[grp] = rng2.permutation(Ke[grp])
            p0 = A0.predict(Ze); p1 = A1.predict(np.concatenate([Ze, Ke], 1)); p2 = A2.predict(np.concatenate([Ze, Ke_sh], 1))
            arms_rows.append({'fold': tag, 'pseudo_target': pr['port'], 'n_eval': len(ye),
                              'enabled_dims': json.dumps(enabled),
                              'A0_acc': float((p0 == ye).mean()), 'A0_ba': ba(ye, p0, C),
                              'A1_acc': float((p1 == ye).mean()), 'A1_ba': ba(ye, p1, C),
                              'A2_acc': float((p2 == ye).mean()), 'A2_ba': ba(ye, p2, C),
                              'rescue_A1_vs_A0': int(((p0 != ye) & (p1 == ye)).sum()),
                              'harm_A1_vs_A0': int(((p0 == ye) & (p1 != ye)).sum())})
            for c in range(C):
                if (ye == c).any():
                    perpair_rows.append({'fold': tag, 'pseudo_target': pr['port'], 'class': vocab[c],
                                         'support': int((ye == c).sum()),
                                         'A0_recall': float((p0[ye == c] == c).mean()),
                                         'A1_recall': float((p1[ye == c] == c).mean()),
                                         'A2_recall': float((p2[ye == c] == c).mean())})
            print('%-10s q=%-10s n=%4d dims=%s | A0 %.3f/%.3f  A1 %.3f/%.3f  A2 %.3f/%.3f  rescue %d harm %d' % (
                tag, pr['port'], len(ye), enabled, arms_rows[-1]['A0_acc'], arms_rows[-1]['A0_ba'],
                arms_rows[-1]['A1_acc'], arms_rows[-1]['A1_ba'], arms_rows[-1]['A2_acc'],
                arms_rows[-1]['A2_ba'], arms_rows[-1]['rescue_A1_vs_A0'], arms_rows[-1]['harm_A1_vs_A0']), flush=True)

    (OUT / 'feature_schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=1))
    (OUT / 'train_and_scaler_lineage.json').write_text(json.dumps(lineage, ensure_ascii=False, indent=1))
    with (OUT / 'source_feature_support.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(support_rows[0].keys())); w.writeheader(); w.writerows(support_rows)
    with (OUT / 'feature_arms_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(arms_rows[0].keys())); w.writeheader(); w.writerows(arms_rows)
    with (OUT / 'per_class_and_per_pair_metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(perpair_rows[0].keys())); w.writeheader(); w.writerows(perpair_rows)
    if arms_rows:
        print('\nA1−A0（BA）：%+.3f ；A2−A0（BA）：%+.3f' % (
            float(np.mean([r['A1_ba'] - r['A0_ba'] for r in arms_rows])),
            float(np.mean([r['A2_ba'] - r['A0_ba'] for r in arms_rows]))))


if __name__ == '__main__':
    main()

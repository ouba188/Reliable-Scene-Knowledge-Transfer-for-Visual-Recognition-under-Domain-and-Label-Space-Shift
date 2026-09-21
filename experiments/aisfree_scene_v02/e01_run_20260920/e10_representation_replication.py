"""E10 (ii) representation replication: is the knowledge increment stable across M0 vs M2 representations?

Protocol (reviewer): on the M0/M2 common pair intersection, use the SAME D+ readout supervision on both
representations. D+ = frozen fit labels + whitelisted A3 additions (M2's augmented label file).
  delta_K(rep) = BA(rep + FULL) - BA(rep + VGM),  FULL=[pca(z),g,a,K], VGM=[pca(z),g,a]
  g = the shared 4-dim geometric/quality feature (m), a = availability mask, K = relations.npz knowledge values.
PCA is fit per representation on that representation's source-fit; g/K/a standardization is computed once on
the shared D+ source-fit (representation-independent) and kept identical across representations.
Evaluation objects/truth/denominator are the frozen E02 pseudo-target pairs.

usage: python e10_representation_replication.py
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
OUT = ROOT / 'e09_attribution'
PCA_DIMS = 64
MIN_VALID, MIN_PORTS, MIN_VAR = 30, 2, 1e-4
PAIRS = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortSaid': 'Port Said'}


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    man = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']
    rel = np.load(REL, allow_pickle=False)
    rel_pos = {str(s): i for i, s in enumerate(rel['sample_id'])}
    rel_phi = rel['phi'].astype(np.float64); rel_avail = rel['available']
    rows = []

    for tag, fold in PAIRS.items():
        run0 = ROOT / 'e01_runs' / (fold + '_rev1')
        run2 = ROOT / 'e01_runs' / (fold + '_M2')
        if not (run2 / 'features.npz').exists():
            print('%-10s 无 M2，跳过' % tag, flush=True)
            continue
        vocab = [str(x) for x in json.loads((run0 / 'heads' / 'class_vocab.json').read_text())]
        vindex = {c: i for i, c in enumerate(vocab)}
        C = len(vocab)
        fitp = set(man[fold].get('source_fit_products', []))
        # D+ readout supervision = M2's augmented label file (frozen + A3)
        dplus = {}
        for r in csv.DictReader((run2 / 'source_fit_labels.csv').open(encoding='utf-8-sig')):
            if r['label'] in vindex:
                dplus[r['sample_id']] = r['label']

        sp = json.loads((E02 / ('source_pseudo_target_splits_%s.json' % tag)).read_text())
        pairs = [p for p in sp['pseudo_targets'] if p.get('assessable') and p['n_eval_labeled'] >= 400]
        if not pairs:
            continue
        # eval labels (frozen object table, same as E02 evaluator)
        eval_lab = {}
        all_prods = set()
        for p in pairs:
            all_prods |= set(p['adapt_products']) | set(p['eval_products'])
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['deployable_offshore'] == '1' and r['product_id'] in all_prods:
                    c = (r.get('ais_final_class') or '').strip()
                    if c in vindex:
                        eval_lab[r['object_id']] = c

        # shared g/K/a standardization on the D+ source-fit (representation-independent)
        feats0 = np.load(run0 / 'features.npz', allow_pickle=False)
        ids0 = feats0['sample_id'].astype(str); m0 = feats0['m'].astype(np.float64)
        sel_d = np.array([i for i, o in enumerate(ids0) if o in dplus])
        dri = np.array([rel_pos.get(o, -1) for o in ids0[sel_d]], dtype=np.int64)
        okr = dri >= 0; sel_d, dri = sel_d[okr], dri[okr]
        Kd_raw = rel_phi[dri]; Ad = rel_avail[dri]
        y_d = np.array([vindex[dplus[ids0[i]]] for i in sel_d])
        enabled = [d for d in range(4) if (Ad[:, d] & np.isfinite(Kd_raw[:, d])).sum() >= MIN_VALID
                   and len(set(feats0['port'].astype(str)[sel_d][Ad[:, d]].tolist())) >= MIN_PORTS
                   and float(Kd_raw[Ad[:, d], d].var()) >= MIN_VAR]
        mean_f = np.array([np.nanmean(Kd_raw[:, d]) for d in enabled])
        std_f = np.array([np.nanstd(Kd_raw[:, d]) for d in enabled]); std_f[std_f < 1e-9] = 1.0
        g_mean = m0[sel_d].mean(0); g_std = m0[sel_d].std(0); g_std[g_std < 1e-9] = 1.0

        def arm_features(rep_dir, mode):
            feats = np.load(rep_dir / 'features.npz', allow_pickle=False)
            z = feats['z']; m = feats['m'].astype(np.float64)
            ids = feats['sample_id'].astype(str); prod = feats['product_id'].astype(str)
            s_fit = np.array([i for i, o in enumerate(ids) if o in dplus])
            fri = np.array([rel_pos.get(o, -1) for o in ids[s_fit]], dtype=np.int64)
            ok = fri >= 0
            s_fit, fri = s_fit[ok], fri[ok]
            yf = np.array([vindex[dplus[ids[i]]] for i in s_fit])
            pca = PCA(n_components=min(PCA_DIMS, z.shape[1], len(s_fit) - 1), random_state=20260920).fit(z[s_fit])
            Zf = pca.transform(z[s_fit]); Gf = (m[s_fit] - g_mean) / g_std
            Af = rel_avail[fri][:, enabled].astype(np.float64)
            Kf = (np.where(np.isfinite(rel_phi[fri][:, enabled]), rel_phi[fri][:, enabled], mean_f) - mean_f) / std_f
            X_vgm = np.concatenate([Zf, Gf, Af], 1); X_full = np.concatenate([X_vgm, Kf], 1)
            vgm = LogisticRegression(max_iter=3000, C=1.0).fit(X_vgm, yf)
            full = LogisticRegression(max_iter=3000, C=1.0).fit(X_full, yf)
            return pca, vgm, full, z, m, ids, prod

        for pr in pairs:
            rec = {'fold': tag, 'pseudo_target': pr['port'], 'n_eval': pr['n_eval_labeled'],
                   'enabled_dims': json.dumps(enabled), 'D_plus_n': int(len(sel_d))}
            for name, rep_dir in (('M0', run0), ('M2', run2)):
                pca, vgm, full, z, m, ids, prod = arm_features(rep_dir, name)
                e_sel = np.array([i for i, o in enumerate(ids) if o in eval_lab
                                  and prod[i] in set(pr['eval_products'])])
                if len(e_sel) < 100:
                    rec['%s_available' % name] = 'no'
                    continue
                ye = np.array([vindex[eval_lab[ids[i]]] for i in e_sel])
                eri = np.array([rel_pos.get(o, -1) for o in ids[e_sel]], dtype=np.int64)
                ok = eri >= 0
                e_sel, eri, ye = e_sel[ok], eri[ok], ye[ok]
                Ze = pca.transform(z[e_sel]); Ge = (m[e_sel] - g_mean) / g_std
                Ae = rel_avail[eri][:, enabled].astype(np.float64)
                Ke = (np.where(np.isfinite(rel_phi[eri][:, enabled]), rel_phi[eri][:, enabled], mean_f) - mean_f) / std_f
                X_vgm_e = np.concatenate([Ze, Ge, Ae], 1); X_full_e = np.concatenate([X_vgm_e, Ke], 1)
                b_vgm = ba(ye, vgm.predict(X_vgm_e), C); b_full = ba(ye, full.predict(X_full_e), C)
                rec['%s_BA_vgm' % name] = b_vgm; rec['%s_BA_full' % name] = b_full
                rec['%s_dK' % name] = b_full - b_vgm
                del z, m
            if rec.get('M2_dK') is not None:
                rows.append(rec)
                print('%-10s q=%-10s n=%4d | M0: VGM %.3f FULL %.3f dK=%+.3f | M2: VGM %.3f FULL %.3f dK=%+.3f' % (
                    tag, pr['port'], rec['n_eval'], rec['M0_BA_vgm'], rec['M0_BA_full'], rec['M0_dK'],
                    rec['M2_BA_vgm'], rec['M2_BA_full'], rec['M2_dK']), flush=True)
        del feats0

    with (OUT / 'representation_replication.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('done; rows=%d' % len(rows))


if __name__ == '__main__':
    main()

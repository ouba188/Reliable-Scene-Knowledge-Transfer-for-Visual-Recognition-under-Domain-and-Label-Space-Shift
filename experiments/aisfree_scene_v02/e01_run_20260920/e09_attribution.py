"""E09 (i) dimension attribution, per the reviewer protocol. CPU-only, no encoder/H/J retrain.

I-B readout ablation (fitted, same PCA/std/supervision/LR C=1):
  V=[z]  VG=[z,g]  VGM=[z,g,a]  FULL=VGM+K_enabled  SINGLE-Dd=VGM+K[d]  DROP-Dd=FULL\K[d]
  g = the 4-dim geometric/quality feature (features.npz 'm'); a = availability mask (enabled dims, 0/1)
  report  dK=BA(FULL)-BA(VGM),  ddrop=BA(FULL)-BA(DROP-Dd),  dsingle=BA(SINGLE-Dd)-BA(VGM)

I-A frozen-model input interventions (the SAME frozen FULL model throughout, no refit):
  source-mean replacement (set standardized dim to 0) | per-dim permutation | full joint permutation
  group = product_id + full enabled-dims availability pattern; pool = ALL eval-product candidates
  (labelled + unlabelled), permute first then score the labelled subset; 20 seeds (20260920..20260939).

Outputs: observed_coverage_by_dimension.csv, dimension_attribution.csv, fixed_model_permutation.csv
usage: python e09_attribution.py
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
SEEDS = list(range(20260920, 20260940))
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
    cov_rows, att_rows, perm_rows = [], [], []

    for tag in FOLDS:
        run = ROOT / 'e01_runs' / (META[tag] + '_rev1')
        vocab = [str(x) for x in json.loads((run / 'heads' / 'class_vocab.json').read_text())]
        vindex = {c: i for i, c in enumerate(vocab)}
        C = len(vocab)
        feats = np.load(run / 'features.npz', allow_pickle=False)
        z = feats['z']; m = feats['m'].astype(np.float64)
        ids = feats['sample_id'].astype(str); prod = feats['product_id'].astype(str)
        fitp = set(man[META[tag]].get('source_fit_products', []))
        lab = {}
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['deployable_offshore'] == '1' and r['product_id'] in fitp:
                    c = (r.get('ais_final_class') or '').strip()
                    if c in vindex:
                        lab[r['object_id']] = c
        sel_f = np.array([i for i, o in enumerate(ids) if o in lab])
        fri = np.array([rel_pos.get(o, -1) for o in ids[sel_f]], dtype=np.int64)
        okr = fri >= 0
        sel_f, fri = sel_f[okr], fri[okr]
        Kf_raw = rel_phi[fri]; Af = rel_avail[fri]
        y_f = np.array([vindex[lab[ids[i]]] for i in sel_f])
        pf = feats['port'].astype(str)[sel_f]

        enabled = []
        for d in range(4):
            mm = Af[:, d] & np.isfinite(Kf_raw[:, d])
            nv = int(mm.sum()); np_ = len(set(pf[mm].tolist()))
            var = float(Kf_raw[mm, d].var()) if nv > 1 else 0.0
            if nv >= MIN_VALID and np_ >= MIN_PORTS and var >= MIN_VAR:
                enabled.append(d)
        if not enabled:
            continue
        mean_f = np.array([np.nanmean(Kf_raw[:, d]) for d in enabled])
        std_f = np.array([np.nanstd(Kf_raw[:, d]) for d in enabled]); std_f[std_f < 1e-9] = 1.0
        g_mean = m[sel_f].mean(0); g_std = m[sel_f].std(0); g_std[g_std < 1e-9] = 1.0
        pca = PCA(n_components=min(PCA_DIMS, z.shape[1], len(sel_f) - 1), random_state=20260920).fit(z[sel_f])
        Zf = pca.transform(z[sel_f]); Gf = (m[sel_f] - g_mean) / g_std
        Kf = (np.where(np.isfinite(Kf_raw[:, enabled]), Kf_raw[:, enabled], mean_f) - mean_f) / std_f
        Afe = Af[:, enabled].astype(np.float64)

        def readout(feat):
            return LogisticRegression(max_iter=3000, C=1.0).fit(feat, y_f)

        # I-B arms fitted on source
        V = readout(Zf)
        VG = readout(np.concatenate([Zf, Gf], 1))
        VGM = readout(np.concatenate([Zf, Gf, Afe], 1))
        FULL = readout(np.concatenate([Zf, Gf, Afe, Kf], 1))
        single = {d: readout(np.concatenate([Zf, Gf, Afe, Kf[:, [enabled.index(d)]]], 1)) for d in enabled}
        drop = {d: readout(np.concatenate([Zf, Gf, Afe, Kf[:, [i for i, e in enumerate(enabled) if e != d]]], 1))
                for d in enabled}

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
            # FULL eval pool = all objects in eval products (labelled + unlabelled), for permutation
            pool = np.array([i for i, p in enumerate(prod.tolist()) if p in set(pr['eval_products'])])
            if len(pool) < 100:
                continue
            pri = np.array([rel_pos.get(o, -1) for o in ids[pool]], dtype=np.int64)
            okp = pri >= 0
            pool, pri = pool[okp], pri[okp]
            labelled = np.array([i for i in pool if ids[i] in lab_e])
            if len(labelled) < 100:
                continue
            ye = np.array([vindex[lab_e[ids[i]]] for i in labelled])
            Ze = pca.transform(z[pool]); Ge = (m[pool] - g_mean) / g_std
            Kp_raw = rel_phi[pri][:, enabled]
            Ke = (np.where(np.isfinite(Kp_raw), Kp_raw, mean_f) - mean_f) / std_f
            Ape = rel_avail[pri][:, enabled].astype(np.float64)
            base = np.concatenate([Ze, Ge, Ape], 1)
            full = np.concatenate([base, Ke], 1)
            labelled_in_pool = np.array([np.where(pool == i)[0][0] for i in labelled])
            li = labelled_in_pool

            # observed coverage per dim (eval objects, pre-imputation valid + non-constant)
            for d in enabled:
                dd = enabled.index(d)
                raw = Kp_raw[:, dd]
                valid = np.isfinite(raw) & rel_avail[pri][:, d]
                cov_rows.append({'fold': tag, 'pseudo_target': pr['port'], 'dim': 'D%d' % d,
                                 'n_eval_candidates': len(pool), 'n_valid': int(valid.sum()),
                                 'n_valid_labeled': int(valid[li].sum()),
                                 'n_nonzero_after_std': int((np.abs(Ke[:, dd]) > 1e-9).sum()),
                                 'n_distinct_values': int(len(np.unique(np.round(Ke[:, dd], 6))))})

            # I-B metrics
            preds = {
                'V': V.predict(Ze), 'VG': VG.predict(np.concatenate([Ze, Ge], 1)),
                'VGM': VGM.predict(base), 'FULL': FULL.predict(full)}
            for d in enabled:
                dd = enabled.index(d)
                preds['SINGLE-D%d' % d] = single[d].predict(np.concatenate([base, Ke[:, [dd]]], 1))
                preds['DROP-D%d' % d] = drop[d].predict(np.concatenate(
                    [base, Ke[:, [i for i, e in enumerate(enabled) if e != d]]], 1))
            ba_v = {k: ba(ye, v[li], C) for k, v in preds.items()}
            dK = ba_v['FULL'] - ba_v['VGM']
            att_rows.append({'fold': tag, 'pseudo_target': pr['port'], 'n_eval': len(labelled),
                             'enabled_dims': json.dumps(enabled), 'BA_FULL': ba_v['FULL'],
                             'BA_VGM': ba_v['VGM'], 'dK_full_vs_vgm': dK,
                             **{'drop_D%d' % d: ba_v['FULL'] - ba_v['DROP-D%d' % d] for d in enabled},
                             **{'single_D%d' % d: ba_v['SINGLE-D%d' % d] - ba_v['VGM'] for d in enabled},
                             'BA_V': ba_v['V'], 'BA_VG': ba_v['VG']})

            # I-A frozen-model interventions on FULL
            def group_key(i):
                return (prod[pool[i]], tuple(rel_avail[pri[i], enabled].tolist()))
            # source-mean replacement
            for d in enabled:
                dd = enabled.index(d)
                rep = full.copy(); rep[:, base.shape[1] + dd] = 0.0
                att_rows[-1]['mean_replace_BA_D%d' % d] = ba(ye, FULL.predict(rep)[li], C)
            # per-dim + full-joint permutation, 20 seeds
            rngs = {s: np.random.default_rng(s) for s in SEEDS}
            for d in enabled:
                dd = enabled.index(d)
                dd_deltas = []
                for s in SEEDS:
                    Kp = full.copy()
                    changed = 0
                    for gk in set(group_key(i) for i in range(len(pool))):
                        idxs = [i for i in range(len(pool)) if group_key(i) == gk]
                        if len(idxs) < 2:
                            continue
                        perm = rngs[s].permutation(len(idxs))
                        col = full[idxs, base.shape[1] + dd]
                        Kp[[idxs[j] for j in range(len(idxs))], base.shape[1] + dd] = col[perm]
                        changed += int((col[perm] != col).sum())
                    dd_deltas.append(ba(ye, FULL.predict(Kp)[li], C) - ba_v['FULL'])
                    perm_rows.append({'fold': tag, 'pseudo_target': pr['port'], 'intervention': 'permute_D%d' % d,
                                      'seed': s, 'changed': changed, 'delta_BA': dd_deltas[-1]})
                att_rows[-1]['permute_D%d_mean_dBA' % d] = float(np.mean(dd_deltas))
                att_rows[-1]['permute_D%d_pct_perturbed' % d] = float(np.mean(
                    [r['changed'] for r in perm_rows if r['fold'] == tag and r['pseudo_target'] == pr['port']
                     and r['intervention'] == 'permute_D%d' % d]) / max(1, len(pool)))
            joint_deltas = []
            for s in SEEDS:
                Kp = full.copy()
                changed = 0
                for gk in set(group_key(i) for i in range(len(pool))):
                    idxs = [i for i in range(len(pool)) if group_key(i) == gk]
                    if len(idxs) < 2:
                        continue
                    perm = rngs[s].permutation(len(idxs))
                    block = full[idxs, base.shape[1]:]
                    Kp[[idxs[j] for j in range(len(idxs))], base.shape[1]:] = block[perm]
                    changed += int(np.any(block[perm] != block, axis=1).sum())
                joint_deltas.append(ba(ye, FULL.predict(Kp)[li], C) - ba_v['FULL'])
                perm_rows.append({'fold': tag, 'pseudo_target': pr['port'], 'intervention': 'permute_joint',
                                  'seed': s, 'changed': changed, 'delta_BA': joint_deltas[-1]})
            att_rows[-1]['permute_joint_mean_dBA'] = float(np.mean(joint_deltas))
            print('%-10s q=%-10s n=%4d dims=%s dK=%+.3f | drop=%s single=%s | joint_perm=%+.3f' % (
                tag, pr['port'], len(labelled), enabled, round(dK, 3),
                {d: round(att_rows[-1]['drop_D%d' % d], 3) for d in enabled},
                {d: round(att_rows[-1]['single_D%d' % d], 3) for d in enabled},
                att_rows[-1]['permute_joint_mean_dBA']), flush=True)

    with (OUT / 'observed_coverage_by_dimension.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cov_rows[0].keys())); w.writeheader(); w.writerows(cov_rows)
    with (OUT / 'dimension_attribution.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(att_rows[0].keys())); w.writeheader(); w.writerows(att_rows)
    with (OUT / 'fixed_model_permutation.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(perm_rows[0].keys())); w.writeheader(); w.writerows(perm_rows)
    print('done')


if __name__ == '__main__':
    main()

"""E06 step 2: M0 vs M1 (same frozen encoder and PCA; M1 adds whitelisted source supervision).

M0: classifier trained on the frozen fit labels only (identical to the Cross baseline).
M1: same encoder/PCA, classifier trained on frozen labels + the whitelisted A3 additions
    (boost_v2 labels on the SAME fit products, objects not already frozen-labelled).

Evaluation: the frozen E02 pseudo-target pairs (same objects, truth version and denominator);
primary metric BA with Acc, per-class recall, rescue/harm reported alongside.

usage: python e06_m0_m1.py
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
E02 = ROOT / 'e02_alignment'
BOOST = ROOT / 'e01_label_boost_v2'
OUT = ROOT / 'e06_source_supervision'
FOLDS = ['Rotterdam', 'Shanghai', 'PortKlang', 'PortSaid']          # additions too small elsewhere
META = {'Rotterdam': 'Rotterdam', 'Shanghai': 'Shanghai', 'PortKlang': 'Port Klang',
        'PortSaid': 'Port Said'}


def ba(y, pred, C):
    rs = [float((pred[y == c] == c).mean()) for c in range(C) if (y == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def main() -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    man = json.loads((KS / 'e01_first_batch' / 'split_manifest.json').read_text())['folds']
    rows = []
    for tag in FOLDS:
        f = man[META[tag]]
        fit_prods = set(f.get('source_fit_products', []))
        run = ROOT / 'e01_runs' / (META[tag] + '_rev1')
        vocab = [str(x) for x in json.loads((run / 'heads' / 'class_vocab.json').read_text())]
        vindex = {c: i for i, c in enumerate(vocab)}
        C = len(vocab)
        feats = np.load(run / 'features.npz', allow_pickle=False)
        z = feats['z']; prod = feats['product_id'].astype(str); ids = feats['sample_id'].astype(str)
        lab = {}
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv_reader(fh):
                if r['deployable_offshore'] == '1' and r['product_id'] in fit_prods:
                    c = (r.get('ais_final_class') or '').strip()
                    if c in vindex:
                        lab[r['object_id']] = c
        sel0 = [i for i, o in enumerate(ids) if o in lab]
        # whitelisted additions: boost_v2 rows on the SAME fit products, in-vocabulary, not already labelled
        add = {}
        for p in sorted(fit_prods):
            bz = BOOST / (p + '.csv.gz')
            if not bz.exists():
                continue
            with gzip.open(bz, 'rt', encoding='utf-8') as fh:
                for r in csv_reader(fh):
                    if r['object_id'] in lab:
                        continue
                    c = r['v2_t1_class'] or r['v2_t2_class']
                    if c in vindex:
                        add[r['object_id']] = c
        sel1 = [i for i, o in enumerate(ids) if o in add]
        print('%-10s 冻结标签 %5d + 白名单新增 %5d（可用于同产品）' % (tag, len(sel0), len(sel1)), flush=True)

        sp = json.loads((E02 / ('source_pseudo_target_splits_%s.json' % tag)).read_text())
        pairs = [p for p in sp['pseudo_targets'] if p.get('assessable') and p['n_eval_labeled'] >= 400]
        if not pairs or not sel1:
            continue
        all_eval_prods = set()
        for pr in pairs:
            all_eval_prods |= set(pr['adapt_products']) | set(pr['eval_products'])
        eval_lab = {}                                   # one table scan for all pairs of this fold
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv_reader(fh):
                if r['deployable_offshore'] == '1' and r['product_id'] in all_eval_prods:
                    c = (r.get('ais_final_class') or '').strip()
                    if c in vindex:
                        eval_lab[r['object_id']] = c
        for pr in pairs:
            lab_e = eval_lab
            sel_a = np.array([i for i, p in enumerate(prod.tolist()) if p in set(pr['adapt_products'])])
            sel_e = np.array([i for i, o in enumerate(ids) if o in lab_e and prod[i] in set(pr['eval_products'])])
            if len(sel_e) < 100 or len(sel_a) < 100:
                continue
            sel0a = np.asarray(sel0, dtype=np.int64)
            pca = PCA(n_components=min(64, z.shape[1], len(sel0a) - 1), random_state=20260920).fit(z[sel0a])
            X0 = pca.transform(z[sel0a]).astype(np.float64)
            y0 = np.array([vindex[lab[ids[i]]] for i in sel0])
            sel01 = np.concatenate([np.asarray(sel0, dtype=np.int64), np.asarray(sel1, dtype=np.int64)])
            X1 = pca.transform(z[sel01]).astype(np.float64)
            y1 = np.concatenate([y0, np.array([vindex[add[ids[i]]] for i in sel1])])
            Xe = pca.transform(z[sel_e]).astype(np.float64)
            ye = np.array([vindex[lab_e[ids[i]]] for i in sel_e])
            m0 = LogisticRegression(max_iter=3000, C=1.0).fit(X0, y0)
            m1 = LogisticRegression(max_iter=3000, C=1.0).fit(X1, y1)
            p0, p1 = m0.predict(Xe), m1.predict(Xe)
            rec = {'fold': tag, 'pseudo_target': pr['port'], 'n_fit_M0': len(sel0), 'n_added': len(sel1),
                   'n_eval': len(sel_e),
                   'M0_acc': float((p0 == ye).mean()), 'M0_ba': ba(ye, p0, C),
                   'M1_acc': float((p1 == ye).mean()), 'M1_ba': ba(ye, p1, C),
                   'rescue_M1_vs_M0': int(((p0 != ye) & (p1 == ye)).sum()),
                   'harm_M1_vs_M0': int(((p0 == ye) & (p1 != ye)).sum()),
                   'per_class_recall_M0': json.dumps({vocab[c]: round(float((p0[ye == c] == c).mean()), 3)
                                                      for c in range(C) if (ye == c).any()}),
                   'per_class_recall_M1': json.dumps({vocab[c]: round(float((p1[ye == c] == c).mean()), 3)
                                                      for c in range(C) if (ye == c).any()}),
                   'per_class_support': json.dumps({vocab[c]: int((ye == c).sum()) for c in range(C) if (ye == c).any()})}
            rows.append(rec)
            print('  %-10s n=%4d | M0 %.3f/%.3f  M1 %.3f/%.3f  rescue %d harm %d' % (
                pr['port'], rec['n_eval'], rec['M0_acc'], rec['M0_ba'], rec['M1_acc'], rec['M1_ba'],
                rec['rescue_M1_vs_M0'], rec['harm_M1_vs_M0']), flush=True)
    if rows:
        import csv as _csv
        with (OUT / 'M0_M1_M2_results.csv').open('w', newline='', encoding='utf-8') as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print('\nM1 − M0（BA）：%+.3f ；Acc：%+.3f' % (
            float(np.mean([r['M1_ba'] - r['M0_ba'] for r in rows])),
            float(np.mean([r['M1_acc'] - r['M0_acc'] for r in rows]))))


def csv_reader(fh):
    import csv
    return csv.DictReader(fh)


if __name__ == '__main__':
    main()

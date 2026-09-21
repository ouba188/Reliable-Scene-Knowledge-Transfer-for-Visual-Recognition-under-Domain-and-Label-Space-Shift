"""Source-side development harness (v3): forward-pass the frozen heads on SOURCE instances.

Source instances = deployable-offshore objects whose product belongs to the fold's source products and
whose object-table ais_final_class is inside the fold's fine vocabulary. Per-head probabilities are
recomputed with the frozen heads via e01_glue_rev1.head_probs (CPU). No target truth is used.

usage: python e01_source_dev.py --fold Shanghai --run /root/autodl-tmp/e01_runs/Shanghai_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
INPUTS = KS / 'e01_inputs'
OUT = ROOT / 'e01_runs' / 'aux_source_dev'
TOOL = KS / 'e01_candidate_ceiling_v1' / 'oracle_audit.py'
for p in (str(INPUTS), str(KS / 'e01_first_batch' / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)


def source_instances(run: Path, feats, source_products: set, vocab: list[str]):
    """indices of source-pool instances carrying a fine label inside the vocabulary."""
    vindex = {c: i for i, c in enumerate(vocab)}
    ids = feats['sample_id'].astype(str)
    keep, y, keep_prod = [], [], []
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        labels = {r['object_id']: r for r in csv.DictReader(fh)
                  if r['deployable_offshore'] == '1' and r['product_id'] in source_products}
    for i, oid in enumerate(ids):
        r = labels.get(oid)
        if r is None:
            continue
        lab = (r.get('ais_final_class') or '').strip()
        if lab in vindex:
            keep.append(i); y.append(vindex[lab]); keep_prod.append(r['product_id'])
    return np.array(keep, dtype=np.int64), np.array(y, dtype=np.int64), keep_prod, len(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=KS / 'e01_first_batch' / 'split_manifest.json')
    a = ap.parse_args()

    fold = json.loads(a.manifest.read_text())['folds'][a.fold]
    src = set(fold.get('source_fit_products', [])) | set(fold.get('source_meta_query_products', [])) | \
        set(fold.get('source_calibration_adapt_products', [])) | set(fold.get('source_calibration_products', []))
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    vocab = json.loads((a.run / 'heads' / 'class_vocab.json').read_text())
    vocab = vocab if isinstance(vocab, list) else list(vocab)

    idx, y, prods, n_source_pool = source_instances(a.run, feats, src, vocab)
    import e01_glue_rev1 as glue
    p, vocab2, _sc = glue.head_probs(a.run, feats, idx)
    p = p.astype(np.float64)
    ids = feats['sample_id'].astype(str)[idx]

    out_base = OUT / (a.fold.replace(' ', '') + '_source')
    out_base.mkdir(parents=True, exist_ok=True)
    npz = OUT / (a.fold.replace(' ', '') + '_source.npz')
    np.savez_compressed(npz, p=p.astype(np.float32), y=y, sample_id=ids.astype('<U128'),
                        class_names=np.array(vocab, dtype='<U64'),
                        product_id=np.array(prods, dtype='<U128'),
                        b1_pred=p.mean(0).argmax(-1).astype(np.int64))
    del feats

    audit_dir = OUT / (a.fold.replace(' ', '') + '_source_audit')   # tool requires a non-existent out dir
    r = subprocess.run([sys.executable, str(TOOL), '--input', str(npz), '--view', 'source', '--out', str(audit_dir)],
                       capture_output=True, text=True)
    rep = audit_dir / 'ORACLE_ONLY_report.json'
    summary = {'fold': a.fold, 'n_source_labeled': int(len(y)), 'source_pool_objects': int(n_source_pool),
               'H': int(p.shape[0]), 'C': len(vocab), 'n_source_products': len(set(prods))}
    if rep.exists():
        d = json.loads(rep.read_text())
        b1 = d['B1']; sh = d['single_head_ORACLE_acc']; fs = d['fixed_subset_ORACLE_acc']
        hp = p.argmax(-1)
        head_ok = np.zeros(len(y), bool)
        for h in range(hp.shape[0]):
            head_ok |= (hp[h] == y)
        b1_pred = p.mean(0).argmax(-1)
        summary.update({'b1_acc': b1.get('acc'), 'b1_ba': b1.get('ba'),
                        'b1_errors': int((b1_pred != y).sum()), 'head_union_acc': float(head_ok.mean()),
                        'errors_rescuable_by_single_head': int(((b1_pred != y) & head_ok).sum()),
                        'errors_no_head_correct': int(((b1_pred != y) & ~head_ok).sum()),
                        'best_single_head_acc': sh.get('acc'), 'best_single_head_ba': sh.get('ba'),
                        'best_single_head_mask': sh.get('subset_mask'),
                        'fixed_subset_acc': fs.get('acc'), 'fixed_subset_ba': fs.get('ba'),
                        'fixed_subset_size': fs.get('size'), 'fixed_subset_mask': fs.get('subset_mask'),
                        'samplewise_acc': (d.get('samplewise_subset_ORACLE') or {}).get('acc'),
                        'mean_pairwise_disagreement': d.get('mean_pairwise_disagreement'),
                        'per_port_BA_by_head': d.get('per_port_BA_by_head')})
        summary['per_class'] = {c: {'n': int((y == i).sum()),
                                    'b1_recall': float((b1_pred[y == i] == i).mean()) if (y == i).any() else None,
                                    'head_union_recall': float(head_ok[y == i].mean()) if (y == i).any() else None}
                                for i, c in enumerate(vocab)}
    else:
        summary['error'] = (r.stderr or 'no report')[-400:]
    (OUT / (a.fold.replace(' ', '') + '_source_summary.json')).write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print('%-10s src_n=%5d/%5d  B1=%.3f/%.3f  head_union=%.3f  1head=%.3f  fixed=%.3f(%s)  err %d/%d 无头可答=%d' % (
        a.fold, summary['n_source_labeled'], summary['source_pool_objects'],
        summary.get('b1_acc', float('nan')), summary.get('b1_ba', float('nan')),
        summary.get('head_union_acc', float('nan')), summary.get('best_single_head_acc', float('nan')),
        summary.get('fixed_subset_acc', float('nan')), summary.get('fixed_subset_mask', '-'),
        summary.get('errors_rescuable_by_single_head', -1), summary.get('b1_errors', -1),
        summary.get('errors_no_head_correct', -1)))


if __name__ == '__main__':
    main()

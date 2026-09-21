"""Evaluation-only, finite candidate-family diagnostics. Never a deployment selector.

Input NPZ (one fold, one fixed truth tier/view, evaluation rows only):
 p: float [H,N,C] (proper class probabilities, parent-summed BEFORE this tool for coarse views)
 y: integer [N] in [0,C); unknown/unresolved/unrepresentable truth is logged by exporter, not coerced
 sample_id: complete unique Unicode [N]; class_names: Unicode [C]
 Optional score: float [H], frozen TARGET-ADAPT min-max scores, NOT fitted using evaluation y
 Optional product_id: Unicode [N]

Enumerates all nonempty equal-weight subsets. H=8 gives 255, fixed per entire fold.
Reports in-sample hindsight ceilings, not generalization or achievable UDA guarantees.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def parent_sum(p, fine_names, mapping):
    """Total, explicit mapping only. Missing labels are errors, not 'other'."""
    names = [str(c) for c in fine_names]
    if len(names) != len(set(names)):
        raise ValueError('duplicate fine class names')
    missing = [c for c in names if c not in mapping or not mapping[c]]
    if missing:
        raise ValueError(f'explicit parent mapping missing for {missing}')
    parents = sorted({str(mapping[c]) for c in names})
    arr = np.asarray(p, dtype=np.float64)
    if arr.shape[-1] != len(names):
        raise ValueError('last axis must match fine_names')
    out = np.stack([arr[..., [i for i,c in enumerate(names) if mapping[c] == a]].sum(axis=-1)
                    for a in parents], axis=-1)
    return out, parents


def validate(p, y, ids, names, scores=None):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    if p.ndim != 3 or not 1 <= p.shape[0] <= 12 or min(p.shape[1:]) < 1:
        raise ValueError('p must have [1..12,N>=1,C>=1] shape')
    H,N,C = p.shape
    if y.shape != (N,) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= C)):
        raise ValueError('y must be supported integer truth [N], 0<=y<C')
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('nonfinite or out-of-range probabilities')
    if not np.allclose(p.sum(axis=-1), 1.0, atol=2e-6, rtol=0):
        raise ValueError('probabilities must sum to 1; do not silently renormalize')
    ids = np.asarray(ids)
    names = np.asarray(names)
    if ids.shape != (N,) or ids.dtype.kind != 'U' or len(set(ids.tolist())) != N:
        raise ValueError('complete unique Unicode sample_id[N] required')
    if names.shape != (C,) or names.dtype.kind != 'U' or len(set(names.tolist())) != C:
        raise ValueError('unique Unicode class_names[C] required')
    if any(not s for s in ids.tolist() + names.tolist()):
        raise ValueError('empty IDs/class names')
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (H,) or not np.all(np.isfinite(scores)):
            raise ValueError('optional frozen adapt scores must be finite [H]')
    return p,y.astype(np.int64),ids,names,scores


def run_audit(p, y, ids, names, scores=None):
    p,y,ids,names,scores = validate(p,y,ids,names,scores)
    H,N,C = p.shape
    counts = np.bincount(y, minlength=C)
    present = np.flatnonzero(counts)
    def metrics(ok):
        rec = {str(names[c]): {'n': int(counts[c]),
               'correct': int(ok[y==c].sum()), 'recall': float(ok[y==c].mean())}
               for c in present}
        return {'acc': float(ok.mean()), 'ba': float(np.mean([v['recall'] for v in rec.values()])),
                'n': N, 'correct': int(ok.sum()), 'per_class': rec}
    def with_delta(ok):
        m = metrics(ok)
        rescue = int((~base_ok & ok).sum()); harm = int((base_ok & ~ok).sum())
        m.update(rescue=rescue, harm=harm, delta_acc=float(ok.mean()-base_ok.mean()),
                 delta_ba=float(m['ba']-base['ba']))
        assert np.isclose(m['delta_acc'], (rescue-harm)/N, atol=1e-12)
        return m
    base_pred = p.mean(axis=0).argmax(axis=-1)
    base_ok = base_pred == y
    base = metrics(base_ok)
    full_mask = (1<<H)-1
    hp = p.argmax(axis=-1)
    h_ok = hp == y[None,:]
    disagree = np.mean(hp[:,None,:] != hp[None,:,:], axis=-1)
    head_rescue = np.sum((~h_ok[:,None,:]) & h_ok[None,:,:], axis=-1)  # row baseline, col alternative
    union_heads = np.any(h_ok, axis=0)
    union_heads_plus_B1 = union_heads | base_ok
    all_argmax_same = np.all(hp == hp[:1,:], axis=0)
    per_object_union = np.zeros(N, bool)
    rows=[]; by_mask={}
    for mask in range(1,1<<H):
        subset=[h for h in range(H) if mask & (1<<h)]
        pred=p[subset].mean(axis=0).argmax(axis=-1)
        ok=pred==y
        per_object_union |= ok
        row={'subset_mask': mask, 'heads': subset, 'size': len(subset), **with_delta(ok)}
        rows.append(row); by_mask[mask]=row
    if not np.allclose([by_mask[full_mask]['acc'],by_mask[full_mask]['ba']], [base['acc'],base['ba']]):
        raise AssertionError('full subset != B1')
    def best(rows, field):
        # Hindsight ties: prefer the unchanged full ensemble, then smaller numeric mask.
        return min(rows, key=lambda r: (-r[field], r['subset_mask'] != full_mask, r['subset_mask']))
    single=[r for r in rows if r['size']==1]
    report={
        'scope': 'EVALUATION-ONLY, empirical ceilings on supplied truth; never deploy oracle-selected subsets',
        'deployable': False, 'uses_evaluation_truth': True,
        'H': H, 'N': N, 'C': C, 'classes': names.tolist(),
        'missing_truth_classes': [str(names[c]) for c in range(C) if counts[c]==0],
        'argmax_tie_rule': 'lowest class index; preserve recorded class order',
        'B1': base,
        'mean_pairwise_disagreement': float(disagree[np.triu_indices(H,1)].mean()) if H>1 else 0.0,
        'pairwise_disagreement': disagree.tolist(),
        'pairwise_head_rescue_counts': head_rescue.tolist(),
        'all_head_argmax_agreement_fraction': float(all_argmax_same.mean()),
        'B1_error_count': int((~base_ok).sum()),
        'single_head_ORACLE_acc': best(single,'acc'),
        'single_head_ORACLE_ba': best(single,'ba'),
        'fixed_subset_ORACLE_acc': best(rows,'acc'),
        'fixed_subset_ORACLE_ba': best(rows,'ba'),
        'samplewise_head_union': metrics(union_heads),
        'samplewise_H_plus_B1_ORACLE': metrics(union_heads_plus_B1),
        'samplewise_subset_ORACLE': metrics(per_object_union),
        'B1_errors_rescuable_by_any_single_head': int((~base_ok & union_heads).sum()),
        'B1_errors_rescuable_by_any_subset': int((~base_ok & per_object_union).sum()),
        'B1_errors_unrescuable_by_equal_weight_subsets': int((~per_object_union).sum()),
        'B1_correct_while_every_single_head_wrong': int((base_ok & ~union_heads).sum()),
        'subset_can_recover_while_every_single_head_wrong': int((per_object_union & ~union_heads).sum()),
        'maximum_rescue_with_zero_harm_fixed_subset': max(r['rescue'] for r in rows if r['harm']==0),
        'n_enumerated_subsets': len(rows)
    }
    if scores is not None:
        # For fixed s_h and scalar tau, feasible subsets are tied-score prefixes.
        # Empty feasible -> fallback to all H. This is a superset of subsets
        # allowed by any extra deployment restrictions such as uninformative tau.
        masks={full_mask}
        for s in np.unique(scores):
            masks.add(sum(1<<int(h) for h in np.flatnonzero(scores<=s)))
        prefixes=[by_mask[m] for m in sorted(masks)]
        report['score_prefix_ORACLE']={
            'frozen_adapt_scores':scores.tolist(),
            'n_distinct_returned_sets':len(prefixes),
            'acc_best':best(prefixes,'acc'), 'ba_best':best(prefixes,'ba'),
            'note':'Hindsight policy-family diagnostic. Does not recommend a tau; no target-label tuning.'}
    return report,rows


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--view',required=True,help='e.g. fine_t1 or coarse_t1_plus_t2')
    args=a.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new audit directory; existing evidence is not overwritten')
    with np.load(args.input,allow_pickle=False) as d:
        required={'p','y','sample_id','class_names'}
        if not required.issubset(d.files): raise ValueError(f'missing {required-set(d.files)}')
        report,rows=run_audit(d['p'],d['y'],d['sample_id'],d['class_names'],d['score'] if 'score' in d.files else None)
        if 'b1_pred' in d.files:
            original = np.asarray(d['b1_pred'])
            recalculated = np.asarray(d['p'], dtype=np.float64).mean(axis=0).argmax(axis=-1)
            if original.shape != recalculated.shape or not np.array_equal(original, recalculated):
                raise ValueError('B1 mismatch: reconcile row order, coarse aggregation, and numerical ties first')
            report['original_B1_prediction_verified'] = True
        else:
            report['original_B1_prediction_verified'] = False
        if 'product_id' in d.files:
            product=d['product_id']
            if product.shape!=(report['N'],) or product.dtype.kind!='U': raise ValueError('bad product_id')
            report['n_distinct_products']=len(set(product.tolist()))
    report['view']=args.view
    report['input_file_sha256']=digest(args.input)
    args.out.mkdir(parents=True)
    (args.out/'ORACLE_ONLY_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    fields=['subset_mask','heads','size','n','correct','acc','ba','rescue','harm','delta_acc','delta_ba']
    with (args.out/'ORACLE_ONLY_subsets.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            v={k:r[k] for k in fields};v['heads']='|'.join(map(str,v['heads']));w.writerow(v)
    print(json.dumps({k:report[k] for k in ('scope','H','N','B1_error_count','n_enumerated_subsets')},ensure_ascii=False))

if __name__=='__main__': main()

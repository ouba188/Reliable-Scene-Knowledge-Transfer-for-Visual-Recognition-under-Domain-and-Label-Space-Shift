"""E01 auxiliary coarse-level evaluation (single-column, does NOT touch models or rules).

For each rev1 fold it projects the fine vocabulary and the evaluator-side true labels onto a coarse
parent taxonomy, then reports:
  * coarse_supported subset: eval instances whose true coarse parent is predictable by the vocab
    (i.e. the coarse parent of at least one vocabulary class)
  * coarse metrics (Acc / BA / per-class) of B0 / B1 / Full on that subset, model predictions
    projected from the fine argmax
  * coverage: how many eval instances (and labels) fall inside vs outside the vocab's parent set

Writes README.md with the required provenance: run dir, prediction/vocabulary hashes, mapping
version, and the explicit statement that this analysis is not used for training or model selection.

usage (server, per fold): python e01_coarse_eval.py --fold Rotterdam --run /root/autodl-tmp/e01_runs/Rotterdam_rev1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
PKG = KS / 'e01_first_batch'
OUT = ROOT / 'e01_runs' / 'aux_coarse_eval_v1'
MAPPING_VERSION = 'coarse_parent_v1'

# fine class -> coarse parent (documented, from the MMSI layer's taxonomy; no target data used)
COARSE = {
    'container_ship': 'cargo', 'general_cargo': 'cargo', 'bulk_carrier': 'cargo',
    'ro_ro_vehicle_carrier': 'cargo', 'refrigerated_cargo': 'cargo', 'cargo_coarse': 'cargo',
    'product_chemical_tanker': 'tanker', 'crude_oil_tanker': 'tanker', 'lpg_lng_tanker': 'tanker',
    'bunker_tanker': 'tanker', 'tanker_coarse': 'tanker',
    'passenger_ship': 'passenger', 'ro_ro_passenger': 'passenger', 'high_speed_passenger': 'passenger',
    'fishing_vessel': 'fishing',
    'tug_towing': 'tug', 'pilot_port_tender': 'tug', 'dredger': 'dredger',
    'offshore_supply': 'offshore', 'offshore_vessel': 'offshore',
    'sailing_vessel': 'pleasure', 'pleasure_craft': 'pleasure', 'yacht': 'pleasure',
    'research_vessel': 'other', 'law_enforcement': 'other', 'military': 'other',
    'ship_untyped': 'untyped', 'untyped': 'untyped', 'unknown': 'untyped', 'non_ship': 'non_ship',
}


def load_meta():
    meta = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                meta[r['object_id']] = r
    return meta


def balanced_accuracy(y, pred):
    vals = [float((pred[y == c] == c).mean()) for c in sorted(set(y)) if (y == c).sum()]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=PKG / 'split_manifest.json')
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    fold = man['folds'][a.fold]
    vocab = json.loads((a.run / 'heads' / 'class_vocab.json').read_text())
    vocab_coarse = sorted({COARSE.get(c, 'other') for c in vocab})
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    probs = np.load(a.run / 'eval_probabilities.npy')          # [H, N_eval, C]
    eidx = np.load(a.run / 'eval_index.npy')
    ret = json.loads((a.run / 'adapt_filter' / 'retention.json').read_text())['retained']
    b0 = json.loads((a.run / 'meta_head_selection.json').read_text())['B0']
    meta = load_meta()

    y_fine = [meta.get(ids[i], {}).get('ais_final_class', '') for i in eidx]
    y_coarse = [COARSE.get(v, '') if v else '' for v in y_fine]
    usable = np.array([1 if v in vocab_coarse else 0 for v in y_coarse], dtype=bool)

    reports, per_rows = {}, []
    for name, heads in [('B0', [b0]), ('B1', list(range(probs.shape[0]))), ('Full', ret)]:
        p = probs[heads].mean(0)
        pred_fine = p.argmax(-1)
        pred_coarse = np.array([COARSE.get(vocab[k], 'other') for k in pred_fine])
        yc = np.array(y_coarse)[usable]
        pc = pred_coarse[usable]
        classes = sorted(set(yc))
        ba_p = [float((pc[yc == c] == c).mean()) for c in classes]
        reports[name] = {
            'n_coarse_supported': int(usable.sum()),
            'acc': float((pc == yc).mean()) if len(yc) else None,
            'ba': float(np.mean(ba_p)) if ba_p else None,
            'per_coarse_class': {c: {'n': int((yc == c).sum()), 'recall': float((pc[yc == c] == c).mean())} for c in classes},
        }
        for c in classes:
            per_rows.append([a.fold, name, c, int((yc == c).sum()), round(float((pc[yc == c] == c).mean()), 4)])

    # coverage over the fixed denominator
    lab_in = np.array([1 if v else 0 for v in y_fine], dtype=bool)
    rec = {
        'fold': a.fold,
        'run': str(a.run),
        'vocabulary_hash': hashlib.sha256(json.dumps(vocab).encode()).hexdigest()[:16],
        'predictions_hash': hashlib.sha256(np.load(a.run / 'eval_probabilities.npy').tobytes()).hexdigest()[:16],
        'features_sha256': json.loads((a.run / 'features.done').read_text())['sha256'][:16],
        'mapping_version': MAPPING_VERSION,
        'coarse_parents_of_vocabulary': vocab_coarse,
        'coverage': {
            'eval_instances': int(len(eidx)),
            'eval_labelled_any': int(lab_in.sum()),
            'coarse_supported': int(usable.sum()),
            'outside_vocab_parents': int(lab_in.sum() - usable.sum()),
            'unlabelled': int(len(eidx) - lab_in.sum()),
        },
        'metrics': reports,
        'note': 'single-column auxiliary analysis; never used for training or model selection',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('coarse_metrics_%s.json' % a.fold)).write_text(json.dumps(rec, indent=1))
    with (OUT / 'coarse_per_class.csv').open('a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['fold', 'bank', 'coarse_class', 'n', 'recall'])
        w.writerows(per_rows)
    print(json.dumps(rec['coverage'], ensure_ascii=False))
    for k, v in reports.items():
        print('  %-4s n=%d acc=%s ba=%s' % (k, v['n_coarse_supported'], v['acc'], v['ba']))


if __name__ == '__main__':
    main()

"""E01 experiment-side glue — revision 1 (2026-09-21), after ChatGPT's execution review.

Fixes vs rev0:
  * B0 anchor selection uses per-port BALANCED accuracy (macro recall over the classes present in
    that port), averaged across meta ports — rev0 averaged per-port plain accuracy.
  * metrics report Acc AND BA and per-class N / recall; rescue/harm use identical denominators.
  * sample_id dtypes widened (<U128) so product suffixes are never truncated.
  * support coverage is reported explicitly: P0 / IO-valid / evaluable / out-of-vocab / unlabelled.

usage: python e01_glue_rev1.py prep|metrics --fold TARGET --run DIR
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
PKG = KS / 'e01_first_batch'
sys.path.insert(0, str(PKG / 'tools'))
from e01_core import aggregate  # noqa: E402

REL = ROOT / 'e01_runs' / 'relations' / 'relations.npz'


def load_meta():
    meta, prod2port = {}, {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                meta[r['object_id']] = r
                prod2port[r['product_id']] = r['port']
    return meta, prod2port


def balanced_accuracy(y_true, y_pred):
    """Macro recall over the classes present in this set (BA)."""
    recalls = []
    for c in sorted(set(y_true)):
        sel = y_true == c
        if sel.sum():
            recalls.append(float((y_pred[sel] == c).mean()))
    return float(np.mean(recalls)) if recalls else None


def per_class_report(y_true, y_pred, vocab):
    out = {}
    for k, c in enumerate(vocab):
        sel = y_true == k
        out[c] = {'n': int(sel.sum()), 'recall': float((y_pred[sel] == k).mean()) if sel.sum() else None}
    return out


def head_probs(run: Path, feats, idx):
    import torch
    from models import CandidateHead
    vocab = json.loads((run / 'heads' / 'class_vocab.json').read_text())
    sc = np.load(run / 'heads' / 'feature_scaler.npz')
    xx = np.concatenate([feats['z'][idx], feats['m'][idx]], 1).astype('float32')
    x = torch.from_numpy((xx - sc['mean']) / sc['std'])
    banks = json.loads((run / 'heads' / 'bank_manifest.json').read_text())
    outs = []
    for rec in banks['heads']:
        # weights_only=False: same toolchain produced these checkpoints (small metadata + tensors).
        ck = torch.load(run / 'heads' / (rec['head_id'] + '.pt'), map_location='cpu', weights_only=False)
        net = CandidateHead(516, len(vocab), rec['architecture'])
        net.load_state_dict(ck['state_dict'])
        net.eval()
        with torch.no_grad():
            outs.append(torch.softmax(net(x), -1).numpy())
    return np.stack(outs), vocab, sc


def rel_means(run: Path, feats, idx, vocab, sc):
    import torch
    from models import RelationMomentModel
    g = np.load(run / 'groups' / 'group_model.npz')
    z16 = ((feats['z'][idx] - g['scaler_mean']) / g['scaler_scale'] - g['pca_mean']) @ g['pca_components'].T
    mm = (feats['m'][idx] - sc['mean'][-4:]) / sc['std'][-4:]
    X = torch.from_numpy(np.concatenate([z16, mm], 1).astype('float32'))
    rman = json.loads((run / 'relations' / 'relation_manifest.json').read_text())
    C, J, D = len(vocab), rman['J'], rman['D']
    out = np.zeros((J, len(idx), C, D), dtype=np.float32)
    for j in range(J):
        # weights_only=False: same toolchain produced checkpoint.
        ck = torch.load(run / 'relations' / ('j%d.pt' % j), map_location='cpu', weights_only=False)
        net = RelationMomentModel(20, C, D)
        net.load_state_dict(ck['state_dict'])
        net.eval()
        with torch.no_grad():
            for c in range(C):
                out[j, :, c, :] = net(X, torch.full((len(idx),), c, dtype=torch.long)).numpy()
    return out, np.array(rman['enabled_dimensions'], dtype=bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prep', 'metrics'])
    ap.add_argument('--fold', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=PKG / 'split_manifest.json')
    ap.add_argument('--config', type=Path, default=PKG / 'config.json')
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    fold = json.loads(a.manifest.read_text())['folds'][a.fold]
    feats = np.load(a.run / 'features.npz', allow_pickle=False)
    rel = np.load(REL, allow_pickle=False)
    ids = feats['sample_id'].astype(str)
    ports = feats['port'].astype(str)
    status = feats['status'].astype(str) if 'status' in feats.files else np.array(['valid'] * len(ids))
    rlook = {s: i for i, s in enumerate(rel['sample_id'].astype(str))}
    meta, prod2port = load_meta()

    def subset(products, only_valid=True):
        ps = set(products)
        out = []
        for i in range(len(ids)):
            if ids[i] not in rlook:
                continue
            if prod2port.get(meta.get(ids[i], {}).get('product_id', ''), '') not in ps and \
               meta.get(ids[i], {}).get('product_id') not in ps:
                continue
            if only_valid and status[i] != 'valid':
                continue
            out.append(i)
        return np.array(out, dtype=np.int64)

    if a.cmd == 'prep':
        probs_all, vocab, sc = head_probs(a.run, feats, np.arange(len(ids)))
        ci = {c: k for k, c in enumerate(vocab)}
        # ---- B0 anchor: per-port BA on source-meta-query, then average across ports ----
        midx = subset(fold['source_meta_query_products'])
        y_lab = np.array([meta[ids[i]]['ais_final_class'] for i in midx])
        mports = np.array([ports[i] for i in midx])
        keep = np.array([k for k, v in enumerate(y_lab) if v in ci])
        midx, y_lab, mports = midx[keep], y_lab[keep], mports[keep]
        y_true = np.array([ci[v] for v in y_lab])
        pred = probs_all[:, midx, :].argmax(-1)
        ba_heads, acc_heads = [], []
        for h in range(probs_all.shape[0]):
            ba_p = [balanced_accuracy(y_true[mports == p], pred[h][mports == p]) for p in sorted(set(mports))]
            ba_p = [x for x in ba_p if x is not None]
            ba_heads.append(float(np.mean(ba_p)) if ba_p else 0.0)
            acc_heads.append(float((pred[h] == y_true).mean()))
        order = sorted(range(len(ba_heads)), key=lambda h: (-ba_heads[h], h))
        b0 = int(order[0])
        json.dump({'per_port_BA_by_head': ba_heads, 'per_port_BA_mean': ba_heads, 'head_Acc': acc_heads,
                   'B0': b0, 'B0_rule': 'max mean per-port BA on source-meta-query, ties -> lowest id',
                   'meta_instances_used': int(len(midx)), 'vocabulary': vocab},
                  (a.run / 'meta_head_selection.json').open('w'), indent=1)
        print('B0 = h%d (跨港 BA 均值 %.4f; 各头 %s)' % (b0, ba_heads[b0], [round(v, 3) for v in ba_heads]))

        aidx = subset(fold['target_adapt_products'])
        eidx = subset(fold['target_eval_products'])
        ridx_a = [rlook[ids[i]] for i in aidx]
        phi = np.nan_to_num(rel['phi'][ridx_a], nan=0.0).astype(np.float32)
        av = rel['available'][ridx_a].astype(bool)
        blocks = rel['spatial_blocks'][ridx_a]
        g = np.load(a.run / 'groups' / 'groups.npz')
        glook = {s: i for i, s in enumerate(g['sample_id'].astype(str))}
        grp = np.array([g['group_id'][glook[ids[i]]] for i in aidx], dtype=np.int64)
        q, enabled = rel_means(a.run, feats, aidx, vocab, sc)
        np.savez_compressed(a.run / 'filter_inputs.npz', phi=phi, available=av, groups=grp,
                            spatial_blocks=blocks, probabilities=probs_all[:, aidx, :].astype(np.float32),
                            relation_means=q, enabled_dimensions=enabled,
                            sample_id=ids[aidx].astype('<U128'), port=ports[aidx].astype('<U64'))
        print('filter_inputs: adapt %d 实例, phi %s, probs %s, q %s' % (len(aidx), phi.shape,
                                                                      probs_all[:, aidx, :].shape, q.shape))

        cal_prod_by_port = defaultdict(list)
        for p in fold['source_calibration_adapt_products']:
            if p in prod2port:
                cal_prod_by_port[prod2port[p]].append(p)
        obs, pred, act, cal_report = [], [], [], {}
        for port in fold['source_calibration_ports']:
            prods = set(cal_prod_by_port.get(port, []))
            if not prods:
                cal_report[port] = {'products': 0, 'instances': 0, 'active_cells': 0}
                continue
            sel = np.array([i for i in range(len(ids)) if ids[i] in rlook and
                            meta.get(ids[i], {}).get('product_id') in prods and status[i] == 'valid'],
                           dtype=np.int64)
            if not len(sel):
                cal_report[port] = {'products': len(prods), 'instances': 0, 'active_cells': 0}
                continue
            ridx = [rlook[ids[i]] for i in sel]
            phi_p = np.nan_to_num(rel['phi'][ridx], nan=0.0).astype(np.float64)
            av_p = rel['available'][ridx].astype(bool)
            bl_p = rel['spatial_blocks'][ridx]
            grp_p = np.array([g['group_id'][glook[ids[i]]] for i in sel], dtype=np.int64)
            q_p, _ = rel_means(a.run, feats, sel, vocab, sc)
            m = aggregate(phi_p, av_p, grp_p, bl_p, probs_all[:, sel, :].astype(np.float64),
                          q_p.astype(np.float64), B=cfg['B'],
                          min_cell=cfg['moment_validity']['min_available_objects_per_cell'],
                          min_blocks=cfg['moment_validity']['min_spatial_blocks_per_cell'],
                          enabled_dimensions=enabled)
            obs.append(m.observed)
            pred.append(m.predicted[b0])
            act.append(m.active)
            cal_report[port] = {'products': len(prods), 'instances': int(len(sel)),
                                'active_cells': int(m.active.sum()),
                                'cell_reasons': dict(Counter(m.reasons[m.reasons != 'active'].ravel().tolist())) if (m.reasons != 'active').any() else {}}
            print('  cal 港 %-16s 产品 %d 实例 %6d 活跃单元 %2d' % (port, len(prods), len(sel), int(m.active.sum())))
        if obs:
            np.savez_compressed(a.run / 'calibration_moments.npz', observed=np.stack(obs),
                                predicted_anchor=np.stack(pred), active=np.stack(act))
        json.dump(cal_report, (a.run / 'calibration_ports_report.json').open('w'), indent=1)
        print('calibration_moments: A=%d' % len(obs))

        np.save(a.run / 'eval_probabilities.npy', probs_all[:, eidx, :].astype(np.float32))
        np.save(a.run / 'eval_index.npy', eidx)
        json.dump({'target_eval_instances': int(len(eidx)), 'target_adapt_instances': int(len(aidx)),
                   'B0': b0, 'H': int(probs_all.shape[0]), 'C': len(vocab)}, (a.run / 'glue_prep.json').open('w'), indent=1)
        print('eval_probabilities: %d 实例' % len(eidx))
        return

    # ---------------- metrics ----------------
    eidx = np.load(a.run / 'eval_index.npy') if (a.run / 'eval_index.npy').exists() else subset(fold['target_eval_products'])
    vocab = json.loads((a.run / 'meta_head_selection.json').read_text())['vocabulary']
    ci = {c: k for k, c in enumerate(vocab)}
    y_all = np.array([meta.get(ids[i], {}).get('ais_final_class', '') for i in eidx])
    probs_all, _, _ = head_probs(a.run, feats, eidx)
    b0 = json.loads((a.run / 'meta_head_selection.json').read_text())['B0']
    ret_file = a.run / 'adapt_filter' / 'retention.json'
    ret = json.loads(ret_file.read_text())['retained'] if ret_file.exists() else list(range(probs_all.shape[0]))
    res = {'n_eval_instances_used': int(len(eidx)), 'B0': b0, 'retained': ret, 'vocabulary': vocab,
           'support': {'p0_offshore_total': 408981, 'eval_valid_instances': int(len(eidx)),
                       'eval_in_vocab_labeled': int(sum(1 for v in y_all if v in ci)),
                       'eval_out_of_vocab_labeled': int(sum(1 for v in y_all if v and v not in ci)),
                       'eval_unlabeled': int(sum(1 for v in y_all if not v)),
                       'note': 'IO/coverage-missing objects stay registered on the P0 side and are '
                               'excluded from this evaluable subset'}}
    keep = np.array([k for k, v in enumerate(y_all) if v in ci])
    y = np.array([ci[y_all[k]] for k in keep])
    pe = np.array([ports[eidx[k]] for k in keep])
    res['evaluable_instances'] = int(len(keep))
    per = {}
    for name, heads in [('B0', [b0]), ('B1', list(range(probs_all.shape[0]))), ('Full', ret)]:
        p = probs_all[heads][:, keep, :].mean(0)
        pred = p.argmax(-1)
        ba_p = [balanced_accuracy(y[pe == q], pred[pe == q]) for q in sorted(set(pe))]
        ba_p = [x for x in ba_p if x is not None]
        per[name] = {'acc': float((pred == y).mean()), 'ba': float(np.mean(ba_p)) if ba_p else None,
                     'n': int(len(y)), 'per_class': per_class_report(y, pred, vocab)}
    res['per'] = per
    p1 = probs_all[list(range(probs_all.shape[0]))][:, keep, :].mean(0).argmax(-1)
    pF = probs_all[ret][:, keep, :].mean(0).argmax(-1)
    res['rescue_vs_B1'] = int(((p1 != y) & (pF == y)).sum())
    res['harm_vs_B1'] = int(((p1 == y) & (pF != y)).sum())
    res['delta_acc_vs_B1'] = (res['rescue_vs_B1'] - res['harm_vs_B1']) / max(1, len(y))
    json.dump(res, (a.run / 'metrics.json').open('w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

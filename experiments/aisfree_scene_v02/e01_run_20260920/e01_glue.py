"""E01 experiment-side glue: model outputs -> package interfaces (and metrics).

Sub-commands
  prep    : build filter_inputs.npz (target adapt), calibration_moments.npz (per calibration port),
            eval_probabilities.npy (target eval), meta_head_selection.json (B0 anchor)
  metrics : compare B0/B1/Full(+retained set) on target eval using the table's AIS labels (evaluator side only)

All arrays are joined by sample_id; no target labels enter any model input.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
PKG = KS / 'e01_first_batch'
sys.path.insert(0, str(PKG / 'tools'))
from e01_core import aggregate  # noqa: E402

REL = ROOT / 'e01_runs' / 'relations' / 'relations.npz'


def load_meta():
    meta = {}
    with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['deployable_offshore'] == '1':
                meta[r['object_id']] = r
    return meta


def head_probs(run: Path, feats, idx):
    """[H,N,C] softmax from the cached heads over the given feature rows."""
    import torch
    sys.path.insert(0, str(PKG / 'tools'))
    from models import CandidateHead
    vocab = json.loads((run / 'heads' / 'class_vocab.json').read_text())
    sc = np.load(run / 'heads' / 'feature_scaler.npz')
    xx = np.concatenate([feats['z'][idx], feats['m'][idx]], 1).astype('float32')
    x = torch.from_numpy((xx - sc['mean']) / sc['std'])
    banks = json.loads((run / 'heads' / 'bank_manifest.json').read_text())
    outs = []
    for rec in banks['heads']:
        # weights_only=False: these checkpoints are produced by this repo's own toolchain and carry
        # small metadata fields (architecture/classes) alongside the tensors.
        ck = torch.load(run / 'heads' / (rec['head_id'] + '.pt'), map_location='cpu', weights_only=False)
        net = CandidateHead(516, len(vocab), rec['architecture'])
        net.load_state_dict(ck['state_dict'])
        net.eval()
        with torch.no_grad():
            outs.append(torch.softmax(net(x), -1).numpy())
    return np.stack(outs), vocab, sc


def rel_means(run: Path, feats, rel, ridx, idx, vocab, sc, devices='cpu'):
    """[J,N,C,D] predicted moments for the given rows (z16 from the frozen group model, m4 scaled)."""
    import torch
    from models import RelationMomentModel
    g = np.load(run / 'groups' / 'group_model.npz')
    z16 = ((feats['z'][idx] - g['scaler_mean']) / g['scaler_scale'] - g['pca_mean']) @ g['pca_components'].T
    mm = (feats['m'][idx] - sc['mean'][-4:]) / sc['std'][-4:]
    inp = np.concatenate([z16, mm], 1).astype('float32')
    X = torch.from_numpy(inp)
    rman = json.loads((run / 'relations' / 'relation_manifest.json').read_text())
    C = len(vocab)
    J = rman['J']
    D = rman['D']
    out = np.zeros((J, len(idx), C, D), dtype=np.float32)
    for j in range(J):
        # weights_only=False: same toolchain-produced checkpoint as above (in_dim/classes/D metadata).
        ck = torch.load(run / 'relations' / ('j%d.pt' % j), map_location='cpu', weights_only=False)
        net = RelationMomentModel(20, C, D)
        net.load_state_dict(ck['state_dict'])
        net.eval()
        with torch.no_grad():
            for c in range(C):
                cid = torch.full((len(idx),), c, dtype=torch.long)
                out[j, :, c, :] = net(X, cid).numpy()
    enabled = np.array(rman['enabled_dimensions'], dtype=bool)
    return out, enabled


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
    lookup = {s: i for i, s in enumerate(ids)}
    rlook = {s: i for i, s in enumerate(rel['sample_id'].astype(str))}
    meta = load_meta()

    def subset(products):
        ps = set(products)
        return np.array([i for i in range(len(ids)) if ids[i] in rlook and
                         (meta.get(ids[i], {}).get('product_id') in ps)], dtype=np.int64)

    if a.cmd == 'prep':
        probs_all, vocab, sc = head_probs(a.run, feats, np.arange(len(ids)))
        # B0 anchor: equal-port balanced accuracy on source-meta-query products
        midx = subset(fold['source_meta_query_products'])
        y = np.array([meta[ids[i]]['ais_final_class'] for i in midx])
        ci = {c: k for k, c in enumerate(vocab)}
        keep = np.array([i for i, v in enumerate(y) if v in ci])
        midx, y = midx[keep], y[keep]
        pred = probs_all[:, midx, :].argmax(-1)
        yy = np.array([ci[v] for v in y])
        mports = np.array([ports[i] for i in midx])
        ba_per_head = []
        for h in range(probs_all.shape[0]):
            accs = []
            for p in sorted(set(mports)):
                sel = mports == p
                if sel.sum() == 0:
                    continue
                accs.append((pred[h][sel] == yy[sel]).mean())
            ba_per_head.append(float(np.mean(accs)))
        b0 = int(np.argmax(ba_per_head))
        json.dump({'equal_port_ba': ba_per_head, 'B0': b0, 'meta_instances': int(len(midx)),
                   'meta_labeled': int(len(midx)), 'vocabulary': vocab},
                  open(a.run / 'meta_head_selection.json', 'w'), indent=1)
        print('B0 = h%d  (equal-port BA %.4f, 其余 %s)' % (b0, ba_per_head[b0],
                                                          [round(v, 3) for v in ba_per_head]))

        aidx = subset(fold['target_adapt_products'])
        eidx = subset(fold['target_eval_products'])
        phi = np.nan_to_num(rel['phi'][[rlook[ids[i]] for i in aidx]], nan=0.0).astype(np.float32)
        av = rel['available'][[rlook[ids[i]] for i in aidx]].astype(bool)
        blocks = rel['spatial_blocks'][[rlook[ids[i]] for i in aidx]]
        g = np.load(a.run / 'groups' / 'groups.npz')
        glook = {s: i for i, s in enumerate(g['sample_id'].astype(str))}
        grp = np.array([g['group_id'][glook[ids[i]]] for i in aidx], dtype=np.int64)
        q, enabled = rel_means(a.run, feats, rel, rlook, aidx, vocab, sc)
        np.savez_compressed(a.run / 'filter_inputs.npz',
                            phi=phi, available=av, groups=grp, spatial_blocks=blocks,
                            probabilities=probs_all[:, aidx, :].astype(np.float32),
                            relation_means=q, enabled_dimensions=enabled,
                            sample_id=ids[aidx].astype('<U64'), port=ports[aidx].astype('<U32'))
        print('filter_inputs: 目标 adapt 实例 %d (phi %s, probs %s, q %s)' % (
            len(aidx), phi.shape, probs_all[:, aidx, :].shape, q.shape))

        # calibration moments per calibration port (adapt products of that port)
        obs, pred, act = [], [], []
        cal_ports = fold['source_calibration_ports']
        cal_prod_by_port = defaultdict(list)
        for p in fold['source_calibration_adapt_products']:
            port = meta.get(p, {}).get('port') or meta.get(next((k for k in meta if k.startswith(p)), ''), {}).get('port')
            r = meta.get(p)
            if r:
                cal_prod_by_port[r['port']].append(p)
        for port in cal_ports:
            prods = set(cal_prod_by_port.get(port, []))
            if not prods:
                continue
            sel = np.array([i for i in range(len(ids)) if ids[i] in rlook and
                            meta.get(ids[i], {}).get('product_id') in prods], dtype=np.int64)
            if not len(sel):
                continue
            phi_p = np.nan_to_num(rel['phi'][[rlook[ids[i]] for i in sel]], nan=0.0).astype(np.float64)
            av_p = rel['available'][[rlook[ids[i]] for i in sel]].astype(bool)
            bl_p = rel['spatial_blocks'][[rlook[ids[i]] for i in sel]]
            grp_p = np.array([g['group_id'][glook[ids[i]]] for i in sel], dtype=np.int64)
            q_p, _ = rel_means(a.run, feats, rel, rlook, sel, vocab, sc)
            m = aggregate(phi_p, av_p, grp_p, bl_p, probs_all[:, sel, :].astype(np.float64),
                          q_p.astype(np.float64), B=cfg['B'],
                          min_cell=cfg['moment_validity']['min_available_objects_per_cell'],
                          min_blocks=cfg['moment_validity']['min_spatial_blocks_per_cell'],
                          enabled_dimensions=enabled)
            obs.append(m.observed)
            pred.append(m.predicted[b0])          # [J,B,D] anchor-only predictions
            act.append(m.active)
            print('  cal 港 %-16s 实例 %6d 活跃单元 %d' % (port, len(sel), int(m.active.sum())))
        np.savez_compressed(a.run / 'calibration_moments.npz',
                            observed=np.stack(obs), predicted_anchor=np.stack(pred), active=np.stack(act))
        print('calibration_moments: A=%d' % len(obs))

        # target eval probabilities for apply_retained
        np.save(a.run / 'eval_probabilities.npy', probs_all[:, eidx, :].astype(np.float32))
        np.save(a.run / 'eval_index.npy', eidx)
        json.dump({'target_eval_instances': int(len(eidx)), 'target_adapt_instances': int(len(aidx)),
                   'B0': b0, 'H': int(probs_all.shape[0]), 'C': len(vocab)},
                  open(a.run / 'glue_prep.json', 'w'), indent=1)
        print('eval_probabilities: %d 实例' % len(eidx))
        return

    # metrics
    eidx = np.load(a.run / 'eval_index.npy') if (a.run / 'eval_index.npy').exists() else subset(fold['target_eval_products'])
    y_true = np.array([meta.get(ids[i], {}).get('ais_final_class', '') for i in eidx])
    # metrics restricted to instances carrying a source vocabulary label (evaluator side)
    vocab = json.loads((a.run / 'meta_head_selection.json').read_text())['vocabulary']
    ci = {c: k for k, c in enumerate(vocab)}
    keep = np.array([i for i, v in enumerate(y_true) if v in ci])
    probs_all, _, _ = head_probs(a.run, feats, eidx)
    b0 = json.loads((a.run / 'meta_head_selection.json').read_text())['B0']
    ret_file = a.run / 'adapt_filter' / 'retention.json'
    ret = json.loads(ret_file.read_text())['retained'] if ret_file.exists() else list(range(probs_all.shape[0]))
    y = np.array([ci[y_true[i]] for i in keep])
    ports_e = np.array([ports[eidx[i]] for i in keep])
    res = {'n_eval_labelled_in_vocab': int(len(keep)), 'n_eval_total': int(len(eidx)),
           'B0': b0, 'retained': ret, 'vocabulary': vocab, 'per': {}}
    for name, heads in [('B0', [b0]), ('B1', list(range(probs_all.shape[0]))), ('Full', ret)]:
        p = probs_all[heads][:, keep, :].mean(0)
        pred = p.argmax(-1)
        acc = float((pred == y).mean())
        accs = []
        for pt in sorted(set(ports_e)):
            s = ports_e == pt
            if s.sum():
                accs.append(float((pred[s] == y[s]).mean()))
        res['per'][name] = {'acc': acc, 'equal_port_acc': float(np.mean(accs)) if accs else None,
                            'n': int(len(y))}
    # rescue / harm vs B1
    p1 = probs_all[list(range(probs_all.shape[0]))][:, keep, :].mean(0).argmax(-1)
    pF = probs_all[ret][:, keep, :].mean(0).argmax(-1)
    res['rescue_vs_B1'] = int(((p1 != y) & (pF == y)).sum())
    res['harm_vs_B1'] = int(((p1 == y) & (pF != y)).sum())
    json.dump(res, open(a.run / 'metrics.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

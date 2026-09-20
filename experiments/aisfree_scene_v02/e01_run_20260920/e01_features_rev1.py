"""E01 per-fold encoder + features — revision 1 (2026-09-21), after ChatGPT's execution review.

Changes vs rev0:
  * sampling: source-fit labelled chips are loaded into RAM (~91 MiB for 2918 x 2 x 128 x 128 uint8)
    and drawn WITH replacement using P(i) = 1/(P_ports * C_p * N_(p,c)) each epoch, so the
    optimiser sees the configured uniform_port_class distribution; rev0's product-blocked
    sequential order was not equivalent and is kept only as the discarded pilot.
  * augmentation: 180-degree rotation p=0.5 and +-4 px translation implemented with reflect
    padding (np.pad(mode='reflect') then crop) - NOT np.roll (which wraps); parameters are drawn
    per (epoch, draw) in the main process and handed to the workers, so nothing depends on
    workers observing an epoch counter.
  * validity: chip-level joint validity from the rev1 shards gates training/export (>=0.95) and
    feeds m4 column 4; sub-threshold objects stay registered with an explicit status.
  * writes are atomic (tmp + rename) with a features.done marker carrying sha256 + row count.

usage: python e01_features_rev1.py --fold Rotterdam [--epochs 30]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
CHIPS = ROOT / 'e01_chips_rev1'
PKG = KS / 'e01_first_batch'
sys.path.insert(0, str(PKG / 'tools'))
from models import SARResNet18GN  # noqa: E402

VALID_GATE = 0.95


def is_fine(cls: str) -> bool:
    return bool(cls) and not cls.endswith('_coarse') and cls not in ('ship_untyped', 'untyped', 'non_ship', 'unknown')


class Shards:
    """Row index over the rev1 shards + table join, without loading chip pixels."""

    def __init__(self, products: set[str]):
        self.meta = {}
        prod2port = {}
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['deployable_offshore'] == '1':
                    self.meta[r['object_id']] = r
                    prod2port[r['product_id']] = r['port']
        self.prod2port = prod2port
        self.rows = []          # (product, row_index_in_shard)
        self.sid = []
        self.port = []
        self.vjoint = []
        for p in sorted(products):
            f = CHIPS / (p + '.npz')
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            ids = z['sample_id']
            vj = z['valid_joint']
            z.close()
            for i, s in enumerate(ids):
                s = str(s)
                self.rows.append((p, i))
                self.sid.append(s)
                self.port.append(self.meta.get(s, {}).get('port', '?'))
                self.vjoint.append(float(vj[i]))
        self.sid = np.array(self.sid, dtype='<U128')
        self.port = np.array(self.port, dtype='<U64')
        self.vjoint = np.array(self.vjoint, dtype=np.float32)
        self._cache = {}

    def chips(self, prod):
        if prod not in self._cache:
            z = np.load(CHIPS / (prod + '.npz'), allow_pickle=False)
            self._cache = {prod: z['chips']}      # one shard at a time
            z.close()
        return self._cache[prod]


class TrainDS(Dataset):
    """(row_index, aug_seed) pairs, so augmentation draws are fixed per epoch in the main process."""

    def __init__(self, sh: Shards, pairs, labels, mean, std):
        self.sh, self.pairs, self.labels, self.mean, self.std = sh, pairs, labels, mean, std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, seed = self.pairs[k]
        p, j = self.sh.rows[i]
        c = self.sh.chips(p)[j].astype(np.float32) / 255.0
        rng = np.random.default_rng(seed)
        if rng.random() < 0.5:                      # 180-degree rotation
            c = c[:, ::-1, ::-1].copy()
        dy, dx = rng.integers(-4, 5, size=2)
        if dy or dx:                                 # reflect-pad translation (no wraparound)
            pad = 4
            c = np.pad(c, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
            c = c[:, pad + int(dy):pad + int(dy) + 128, pad + int(dx):pad + int(dx) + 128].copy()
        c = (c - self.mean) / self.std
        return torch.from_numpy(c), torch.tensor(int(self.labels[i]), dtype=torch.long)


def build_vocab(sh: Shards, src_products: set[str], src_ports: set[str]):
    per_class = defaultdict(lambda: defaultdict(int))
    for i, sid in enumerate(sh.sid):
        m = sh.meta.get(str(sid))
        p, _ = sh.rows[i]
        if not m or p not in src_products or sh.port[i] not in src_ports:
            continue
        if m.get('ais_class_level') != 'fine' or sh.vjoint[i] < VALID_GATE:
            continue
        cls = m.get('ais_final_class') or ''
        if is_fine(cls):
            per_class[cls][sh.port[i]] += 1
    return {c: dict(ports) for c, ports in per_class.items()
            if sum(ports.values()) >= 100 and len(ports) >= 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--manifest', type=Path, default=PKG / 'split_manifest.json')
    ap.add_argument('--config', type=Path, default=PKG / 'config.json')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    fold = json.loads(a.manifest.read_text())['folds'][a.fold]
    run_seed = cfg['seeds']['first_run']
    out = a.out or (ROOT / 'e01_runs' / (a.fold + '_rev1'))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'features.done').unlink(missing_ok=True)

    needed = set(fold['source_fit_products']) | set(fold['source_meta_query_products']) | \
        set(fold['source_calibration_products']) | set(fold['target_adapt_products']) | set(fold['target_eval_products'])
    src_products = set(fold['source_fit_products'])
    src_ports = set(fold['source_fit_ports'])
    sh = Shards(needed)
    print('实例总数 %d (产品 %d)' % (len(sh.rows), len({sh.rows[i][0] for i in range(len(sh.rows))})), flush=True)
    gate = sh.vjoint >= VALID_GATE
    print('联合有效>=0.95 的实例: %d / %d (%.1f%%)' % (gate.sum(), len(gate), 100 * gate.mean()), flush=True)

    vocab = build_vocab(sh, src_products, src_ports)
    cls_list = sorted(vocab)
    ci = {c: k for k, c in enumerate(cls_list)}
    print('词表 %d 类: %s' % (len(cls_list), cls_list), flush=True)
    json.dump({'vocabulary': {c: vocab[c] for c in cls_list}, 'valid_gate': VALID_GATE,
               'source_fit_ports': sorted(src_ports)},
              (out / 'vocabulary.json').open('w', encoding='utf-8'), ensure_ascii=False, indent=1)

    labels = np.full(len(sh.rows), -1, dtype=np.int64)
    for i, sid in enumerate(sh.sid):
        m = sh.meta.get(str(sid))
        p, _ = sh.rows[i]
        if m and p in src_products and sh.port[i] in src_ports and gate[i] and m.get('ais_class_level') == 'fine':
            c = m.get('ais_final_class') or ''
            if c in ci:
                labels[i] = ci[c]
    train_idx = np.where(labels >= 0)[0]
    print('源 fit 有标签且有效实例 %d（词表内）' % len(train_idx), flush=True)

    # balanced sampling weights P(i) = 1/(P * C_p * N_(p,c)) over the labelled source-fit pool
    ports_u = sorted({sh.port[i] for i in train_idx})
    P = len(ports_u)
    n_by = Counter((sh.port[i], int(labels[i])) for i in train_idx)
    nclass_by_port = defaultdict(set)
    for (pt, cc) in n_by:
        nclass_by_port[pt].add(cc)
    w = np.array([1.0 / (P * len(nclass_by_port[sh.port[i]]) * n_by[(sh.port[i], int(labels[i]))]) for i in train_idx])
    w = w / w.sum()
    print('采样权重: %d 港, 最大 %.6f 最小 %.6f' % (P, w.max(), w.min()), flush=True)

    # load the labelled chips into RAM (~91 MiB for ~3k chips)
    RAM = np.zeros((len(train_idx), 2, 128, 128), dtype=np.uint8)
    for k, i in enumerate(train_idx):
        p, j = sh.rows[i]
        RAM[k] = sh.chips(p)[j]
    print('训练池进内存: %s = %.1f MB' % (RAM.shape, RAM.nbytes / 2**20), flush=True)
    mu = RAM.reshape(len(RAM), 2, -1).mean(axis=(0, 2)) / 255.0
    sd = RAM.reshape(len(RAM), 2, -1).std(axis=(0, 2)) / 255.0 + 1e-6
    mean = mu.reshape(2, 1, 1).astype(np.float32)
    std = sd.reshape(2, 1, 1).astype(np.float32)
    np.savez(out / 'norm_stats.npz', mean=mu.astype(np.float32), std=sd.astype(np.float32),
             n_chips=len(RAM), seed=run_seed)
    print('归一化（source-fit 有效 chip）: VV %.4f/%.4f  VH %.4f/%.4f' % (mu[0], sd[0], mu[1], sd[1]), flush=True)

    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model = SARResNet18GN(num_classes=len(cls_list)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    crit = nn.CrossEntropyLoss()
    steps_per_epoch = max(1, math.ceil(len(train_idx) / (a.batch * a.accum)))
    warm = 3 * steps_per_epoch
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    rng = np.random.default_rng(run_seed)
    draw_log = []
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        sel = rng.choice(len(train_idx), size=len(train_idx), replace=True, p=w)
        draw_log.append({'epoch': ep + 1, 'ports': dict(Counter(sh.port[train_idx[s]] for s in sel).most_common(8)),
                         'classes': dict(Counter(int(labels[train_idx[s]]) for s in sel)),
                         'draw_seed': int(rng.integers(2**31))})
        pairs = [(int(train_idx[s]), int(rng.integers(2**31))) for s in sel]
        ds = TrainDS(sh, pairs, labels, mean, std)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        pin_memory=(device.type == 'cuda'))
        model.train()
        opt.zero_grad(set_to_none=True)
        tot = nb = 0
        for bidx, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), y) / a.accum
            loss.backward()
            tot += float(loss.detach()) * a.accum
            nb += 1
            if (bidx + 1) % a.accum == 0 or bidx == len(dl) - 1:
                if step < warm:
                    lr = a.lr * (step + 1) / warm
                else:
                    t = (step - warm) / max(1, a.epochs * steps_per_epoch - warm)
                    lr = 3e-6 + 0.5 * (a.lr - 3e-6) * (1 + math.cos(math.pi * min(1.0, t)))
                for g in opt.param_groups:
                    g['lr'] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        print('  epoch %2d/%d loss %.4f lr %.2e (%.0fs)' % (ep + 1, a.epochs, tot / max(1, nb), lr, time.time() - t0), flush=True)
    torch.save({'model': model.state_dict(), 'classes': cls_list, 'seed': run_seed,
                'epochs': a.epochs, 'amp_bf16': use_amp, 'valid_gate': VALID_GATE},
               out / 'encoder_last.pt')
    json.dump(draw_log, (out / 'epoch_draws.json').open('w'), indent=1)
    print('encoder_last.pt 已保存', flush=True)

    # frozen encode (no augmentation) using the RAM pool for efficiency on labelled rows, shard-wise
    # for the rest
    model.eval()
    N = len(sh.rows)
    Z = np.zeros((N, 512), dtype=np.float32)
    with torch.no_grad():
        for p in sorted({sh.rows[i][0] for i in range(N)}):
            idxs = [i for i in range(N) if sh.rows[i][0] == p]
            if not idxs:
                continue
            arr = sh.chips(p)
            for start in range(0, len(idxs), 256):
                chunk = idxs[start:start + 256]
                batch = np.stack([arr[sh.rows[i][1]] for i in chunk]).astype(np.float32) / 255.0
                batch = (batch - mean) / std
                x = torch.from_numpy(batch).to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    z = model.encode(x)
                Z[chunk] = z.float().cpu().numpy()
            if (sorted({sh.rows[i][0] for i in range(N)}).index(p) + 1) % 40 == 0:
                print('  编码至产品 %d' % (sorted({sh.rows[i][0] for i in range(N)}).index(p) + 1), flush=True)
    M = np.zeros((N, 4), dtype=np.float32)
    for i, sid in enumerate(sh.sid):
        r = sh.meta.get(str(sid), {})
        pl = float(r.get('obb_long_px') or 1) * 10.0
        ps = float(r.get('obb_short_px') or 1) * 10.0
        M[i] = [math.log1p(pl), math.log1p(ps), math.log(max(pl, 1e-3) / max(ps, 1e-3)), float(sh.vjoint[i])]
    tmp = out / 'features.npz.tmp'
    # np.savez appends .npz to a path lacking that suffix, so write through a file object
    with open(tmp, 'wb') as fh:
        np.savez(fh, z=Z, m=M, sample_id=sh.sid.astype('<U128'),
                 product_id=np.array([sh.rows[i][0] for i in range(N)], dtype='<U128'),
                 port=sh.port.astype('<U64'),
                 valid_joint=sh.vjoint.astype(np.float32), status=np.where(gate, 'valid', 'io_coverage_missing'))
    h = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.rename(out / 'features.npz')
    with (out / 'source_fit_labels.csv').open('w', newline='', encoding='utf-8') as fh:
        w2 = csv.writer(fh)
        w2.writerow(['sample_id', 'label'])
        for i in train_idx:
            w2.writerow([sh.sid[i], cls_list[labels[i]]])
    (out / 'features.done').write_text(json.dumps({'rows': N, 'sha256': h, 'labelled': int(len(train_idx)),
                                                   'classes': len(cls_list), 'valid_gate': VALID_GATE,
                                                   'epochs': a.epochs, 'seconds': round(time.time() - t0, 1)}) + '\n')
    print('features.npz 完成 %d 行 sha256 %s' % (N, h[:16]), flush=True)


if __name__ == '__main__':
    main()

"""E01 per-fold source encoder training + feature/relation export (server side).

Implements exactly the package spec (config.json / docs/CANDIDATE_BANK.md):
  - SARResNet18GN (tools/models.py), 2ch, 30 epochs, AdamW 3e-4 / WD 1e-4,
    microbatch 64 x accum 2, 3-epoch warmup + cosine (min_lr 3e-6), grad clip 5.0,
    last-epoch checkpoint, aug = 180-deg rotation p=0.5 +-4px translation (VV/VH synced, nothing else)
  - uint8/255 then source-fit-only per-channel mean/std
  - label vocabulary from source-fit fine classes: >=100 instances and >=3 source-fit ports
  - frozen encoder -> z512 (no aug) for every needed instance
  - features.npz: z[N,512], m[N,4], sample_id/product_id/port  (strings, no object pickle)
  - labels csv: sample_id,label  (source-fit only; target labels never read here)

usage:
  python e01_features.py --fold Rotterdam [--epochs 30] [--limit-chips N]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/root/autodl-tmp')
KS = ROOT / 'knowledge_841_20260920'
CHIPS = ROOT / 'e01_chips'
PKG = KS / 'e01_first_batch'

sys.path.insert(0, str(PKG / 'tools'))
from models import SARResNet18GN  # noqa: E402


class ChipStore:
    """Lazy per-product chip shards + table join.

    npz access decompresses the whole array each time, so cache the decoded arrays with a small
    LRU (per-product iteration keeps this at 1-2 entries; training random access stays cheap).
    """

    def __init__(self, products: set[str], cache_size: int = 6):
        self.shards = {}
        for p in sorted(products):
            f = CHIPS / (p + '.npz')
            if f.exists():
                self.shards[p] = f
        self.meta = {}
        with gzip.open(KS / 'objects' / 'objects_dedup_mask.csv.gz', 'rt', encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                if r['deployable_offshore'] != '1':
                    continue
                self.meta[r['object_id']] = r
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def sample_ids(self, prod):
        """Cheap accessor: only the sample_id array (npz stores members separately)."""
        if not hasattr(self, '_sid_cache'):
            self._sid_cache = {}
        if prod not in self._sid_cache:
            z = np.load(self.shards[prod], allow_pickle=False)
            self._sid_cache[prod] = z['sample_id']
            z.close()
        return self._sid_cache[prod]

    def shard(self, prod):
        hit = self._cache.get(prod)
        if hit is not None:
            self._cache.move_to_end(prod)
            return hit
        z = np.load(self.shards[prod], allow_pickle=False)
        arr = {'chips': z['chips'], 'valid_fraction': z['valid_fraction'],
               'clipped': z['clipped'], 'sample_id': z['sample_id']}
        z.close()
        self._cache[prod] = arr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr


class FoldData:
    def __init__(self, store: ChipStore, products: list[str]):
        self.store = store
        self.rows = []          # (product, row_index)
        self.port = []
        self.sid = []
        for p in products:
            if p not in store.shards:
                continue
            for i, sid in enumerate(store.sample_ids(p)):
                self.rows.append((p, i))
                self.sid.append(str(sid))
                m = store.meta.get(str(sid))
                self.port.append(m['port'] if m else '?')
        self.sid = np.array(self.sid)

    def m4(self, i):
        r = self.store.meta.get(str(self.sid[i]))
        pl = float(r['obb_long_px']) * 10.0
        ps = float(r['obb_short_px']) * 10.0
        vf = float(r['valid_fraction'] or 0)
        return [math.log1p(pl), math.log1p(ps), math.log(max(pl, 1e-3) / max(ps, 1e-3)), vf]

    def chip(self, i):
        p, j = self.rows[i]
        return self.store.shard(p)['chips'][j]


class TrainDS(Dataset):
    def __init__(self, fd: FoldData, idx, labels, mean, std, augment: bool, seed: int):
        self.fd, self.idx, self.labels = fd, idx, labels
        self.mean, self.std, self.augment, self.seed = mean, std, augment, seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        # batch_sampler yields global row indices into fd.rows
        c = self.fd.chip(int(i)).astype(np.float32) / 255.0
        if self.augment:
            rng = np.random.default_rng((self.seed + int(i)) % (2**31))
            if rng.random() < 0.5:
                c = c[:, ::-1, ::-1].copy()          # 180-deg rotation
            dy, dx = rng.integers(-4, 5, size=2)
            if dy or dx:
                c = np.roll(np.roll(c, int(dy), axis=1), int(dx), axis=2)
        c = (c - self.mean) / self.std
        return torch.from_numpy(c), torch.tensor(int(self.labels[int(i)]), dtype=torch.long)


def is_fine(cls: str) -> bool:
    """Fine vocabulary: level 'fine' AND not a coarse bucket (matches E00 fine_class_resolved)."""
    return bool(cls) and not cls.endswith('_coarse') and cls != 'ship_untyped'


def build_vocab(fd: FoldData, source_ports: set[str]):
    """fine classes with >=100 instances and >=3 source-fit ports."""
    per_class = defaultdict(lambda: defaultdict(int))
    for i, sid in enumerate(fd.sid):
        m = fd.store.meta.get(str(sid))
        if not m or m.get('ais_class_level') != 'fine' or fd.port[i] not in source_ports:
            continue
        cls = m.get('ais_final_class') or ''
        if is_fine(cls):
            per_class[cls][fd.port[i]] += 1
    vocab = {}
    for cls, ports in sorted(per_class.items()):
        n = sum(ports.values())
        if n >= 100 and len(ports) >= 3:
            vocab[cls] = n
    return vocab


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
    ap.add_argument('--limit-chips', type=int, help='smoke: cap source-fit instances')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding='utf-8'))
    man = json.load(open(args.manifest, encoding='utf-8'))
    fold = man['folds'][args.fold]
    run_seed = cfg['seeds']['first_run']
    out = args.out or (ROOT / 'e01_runs' / args.fold)
    out.mkdir(parents=True, exist_ok=True)

    src_ports = set(fold['source_fit_ports'])
    needed = set(fold['source_fit_products']) | set(fold['source_meta_query_products']) | \
        set(fold['source_calibration_products']) | set(fold['target_adapt_products']) | set(fold['target_eval_products'])
    print('折 %s: 源fit %d 产品, meta %d, cal %d, 目标 adapt %d + eval %d' % (
        args.fold, len(fold['source_fit_products']), len(fold['source_meta_query_products']),
        len(fold['source_calibration_products']), len(fold['target_adapt_products']), len(fold['target_eval_products'])), flush=True)

    store = ChipStore(needed)
    print('可用分片 %d / %d 产品' % (len(store.shards), len(needed)), flush=True)
    fd = FoldData(store, sorted(needed))
    print('实例总数 %d' % len(fd.rows), flush=True)

    vocab = build_vocab(fd, src_ports)
    print('源类别词表: %d 类 -> %s' % (len(vocab), list(vocab)[:8]), flush=True)
    json.dump({'vocabulary': vocab, 'source_fit_ports': sorted(src_ports)},
              open(out / 'vocabulary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    cls_list = sorted(vocab)
    cls_index = {c: k for k, c in enumerate(cls_list)}
    labels = np.full(len(fd.rows), -1, dtype=np.int64)
    src_products = set(fold['source_fit_products'])
    for i, sid in enumerate(fd.sid):
        m = store.meta.get(str(sid))
        p, _j = fd.rows[i]
        if m and p in src_products and m.get('ais_class_level') == 'fine':
            c = m.get('ais_final_class') or ''
            if c in cls_index:
                labels[i] = cls_index[c]
    train_idx = np.where(labels >= 0)[0]
    if args.limit_chips:
        train_idx = train_idx[:args.limit_chips]
    print('源fit 有标签实例 %d（词表内）' % len(train_idx), flush=True)

    # source-fit-only per-channel mean/std (stratified over products: random index access
    # would decompress one shard per sample)
    by_prod = defaultdict(list)
    for i in train_idx:
        by_prod[fd.rows[i][0]].append(i)
    rng0 = np.random.default_rng(run_seed)
    prods_norm = sorted(by_prod)
    rng0.shuffle(prods_norm)
    prods_norm = prods_norm[:40]
    sub = []
    for p in prods_norm:
        take = by_prod[p]
        sub += list(rng0.choice(take, size=min(10, len(take)), replace=False))
    s1 = np.zeros(2, dtype=np.float64)
    s2 = np.zeros(2, dtype=np.float64)
    n = 0
    for i in sub:
        c = fd.chip(i).astype(np.float64) / 255.0
        s1 += c.mean(axis=(1, 2))
        s2 += (c ** 2).mean(axis=(1, 2))
        n += 1
    m_ = s1 / max(1, n)
    v_ = np.maximum(s2 / max(1, n) - m_ ** 2, 0.0)
    mean = m_.reshape(2, 1, 1).astype(np.float32)
    std = (np.sqrt(v_).reshape(2, 1, 1) + 1e-6).astype(np.float32)
    np.savez(out / 'norm_stats.npz', mean=mean.reshape(2), std=std.reshape(2))
    print('归一化 mean', mean.reshape(2).round(4), 'std', std.reshape(2).round(4), flush=True)

    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('设备:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')

    ds = TrainDS(fd, train_idx, labels, mean, std, augment=True, seed=run_seed)

    def product_batches(indices, batch, seed):
        """Shuffle products, then emit batches of chips from one product at a time.

        Keeps a single ~30 MB shard resident instead of loading one per sample.
        """
        rng = np.random.default_rng(seed)
        groups = defaultdict(list)
        for i in indices:
            groups[fd.rows[i][0]].append(i)
        order = sorted(groups)
        rng.shuffle(order)
        buf = []
        for p in order:
            idxs = groups[p]
            rng.shuffle(idxs)
            buf += idxs
            while len(buf) >= batch:
                yield buf[:batch]
                buf = buf[batch:]
        if buf:
            yield buf

    class ProductBatchSampler:
        """Reseeds each epoch so product order / within-product order change (still one shard at a time)."""

        def __init__(self, indices, batch, seed):
            self.indices, self.batch, self.seed, self.epoch = indices, batch, seed, 0

        def __iter__(self):
            self.epoch += 1
            yield from product_batches(self.indices, self.batch, self.seed + 7919 * self.epoch)

        def __len__(self):
            return max(1, math.ceil(len(self.indices) / self.batch))

    dl = DataLoader(ds, batch_sampler=ProductBatchSampler(train_idx, args.batch, run_seed),
                    num_workers=args.workers, pin_memory=(device.type == 'cuda'),
                    persistent_workers=args.workers > 0)
    model = SARResNet18GN(num_classes=len(cls_list)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, math.ceil(len(ds) / (args.batch * args.accum)))
    warm = 3 * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        t = (step - warm) / max(1, 30 * steps_per_epoch - warm)
        return 3e-6 + 0.5 * (args.lr - 3e-6) * (1 + math.cos(math.pi * min(1.0, t)))

    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    crit = nn.CrossEntropyLoss()
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        nb = 0
        opt.zero_grad(set_to_none=True)
        for bidx, (x, y) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y) / args.accum
            loss.backward()
            tot += float(loss) * args.accum
            nb += 1
            if (bidx + 1) % args.accum == 0 or bidx == len(dl) - 1:
                for g in opt.param_groups:
                    g['lr'] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        print('  epoch %2d/%d  loss %.4f  lr %.2e  (%.0fs)' % (ep + 1, args.epochs, tot / max(1, nb),
                                                              opt.param_groups[0]['lr'], time.time() - t0), flush=True)
    torch.save({'model': model.state_dict(), 'classes': cls_list, 'seed': run_seed,
                'epochs': args.epochs, 'amp_bf16': use_amp}, out / 'encoder_last.pt')
    print('已保存 encoder_last.pt', flush=True)

    # frozen encode (no augmentation) for all needed instances
    model.eval()
    Z = np.zeros((len(fd.rows), 512), dtype=np.float32)
    M = np.zeros((len(fd.rows), 4), dtype=np.float32)
    with torch.no_grad():
        buf = []
        buf_i = []
        for i in range(len(fd.rows)):
            c = fd.chip(i).astype(np.float32) / 255.0
            c = (c - mean) / std
            buf.append(c)
            buf_i.append(i)
            M[i] = fd.m4(i)
            if len(buf) >= 256 or i == len(fd.rows) - 1:
                x = torch.from_numpy(np.stack(buf)).to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    z = model.encode(x)
                Z[buf_i] = z.float().cpu().numpy()
                buf, buf_i = [], []
                if (i + 1) % 20000 < 256:
                    print('  编码 %d/%d' % (i + 1, len(fd.rows)), flush=True)
    ports = np.array(fd.port, dtype='<U32')
    prods = np.array([fd.rows[i][0] for i in range(len(fd.rows))], dtype='<U64')
    np.savez(out / 'features.npz', z=Z, m=M, sample_id=fd.sid.astype('<U64'),
             product_id=prods, port=ports)
    print('features.npz 写出: z%s m%s' % (Z.shape, M.shape), flush=True)

    with (out / 'source_fit_labels.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['sample_id', 'label'])
        for i in train_idx:
            w.writerow([fd.sid[i], cls_list[labels[i]]])
    print('source_fit_labels.csv 写出 %d 行' % len(train_idx), flush=True)
    json.dump({'fold': args.fold, 'instances': len(fd.rows), 'source_fit_labelled': int(len(train_idx)),
               'classes': len(cls_list), 'epochs': args.epochs, 'amp_bf16': use_amp,
               'seconds': round(time.time() - t0, 1)},
              open(out / 'run_info.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()

"""e35: fine-tuned visual baseline (the decisive test).

Per held-out port p: fine-tune an ImageNet ResNet50 on the SOURCE ports only (chips from the uint8
cache), then re-run V / V+K with the fine-tuned features. Question: does the knowledge's gain shrink
once the visual model is actually trained on the task?

usage: python e35_finetune_baseline.py [n_folds] [epochs] [--full]
"""
import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
CACHE = Path(r'E:/临时会话/visual_reliable_baseline/chip_cache')
OUTF = Path(r'E:/临时会话/visual_reliable_baseline/features_ft')
OUTF.mkdir(parents=True, exist_ok=True)
LEGAL = list(range(0, 71))
ALPHAS = [0.1, 0.3, 1.0]

rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
n = len(rows); C = int(y.max()) + 1
PLIST = sorted(set(ports.tolist()))
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
Kp = np.zeros_like(K)                     # 港内百分位（e45：目前最优知识配置）
for _j, _pj in enumerate(PLIST):
    _m = ports == _pj
    _n = int(_m.sum())
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, _n - 1)
VV = np.load(CACHE / 'images_uint8_vv.npy', mmap_mode='r')
VH = np.load(CACHE / 'images_uint8_vh.npy', mmap_mode='r')


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def batch(idx, dev, train=False, gen=None):
    a = np.asarray(VV[idx], np.float32)[:, 0]        # (B,128,128)
    b = np.asarray(VH[idx], np.float32)[:, 0]
    x = np.stack([a, b], 1)
    x = (x / 255.0 - 0.5) / 0.25
    if train and gen is not None:
        if gen.random() < 0.5:
            x = x[:, :, :, ::-1]
        if gen.random() < 0.5:
            x = x[:, :, ::-1, :]
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev)


def finetune(tr, epochs, dev):
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
    with torch.no_grad():                            # 2 通道：ImageNet 3 通道均值折叠
        w = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).conv1.weight.data
        net.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, 2, 1, 1) * 1.5
    net.fc = nn.Linear(2048, C)
    net = net.to(dev)      # float32 (batch is 128x128; 8GB is plenty)
    for _n, _pr in net.named_parameters():        # 只训 layer3/layer4/分类头（标准配方，快 3-4 倍）
        _pr.requires_grad = _n.startswith(('layer3', 'layer4', 'fc'))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    gen = np.random.default_rng(0)
    B = 32      # frozen early layers free the memory for a real batch: this box runs at <2 GB free: Windows WDDM makes large CUDA allocations flaky here
    for ep in range(epochs):
        perm = gen.permutation(len(tr))
        net.train(); tot = 0.0
        for i in range(0, len(perm), B):
            sub = tr[perm[i:i + B]]
            x = batch(sub, dev, train=True, gen=gen)
            t = torch.from_numpy(y[sub]).to(dev)
            opt.zero_grad()
            out = net(x).float()
            loss = lossf(out, t)
            loss.backward(); opt.step()
            tot += float(loss) * len(sub)
        print('    ep%d loss %.4f' % (ep + 1, tot / max(1, len(tr))), flush=True)
    net.eval()
    return net


@torch.no_grad()
def extract(net, idx, dev):
    net.eval()
    out = []
    B = 256
    for i in range(0, len(idx), B):
        x = batch(idx[i:i + B], dev)
        f = net.avgpool(net.layer4(net.layer3(net.layer2(net.layer1(
            net.relu(net.bn1(net.conv1(x))))))))          # 2048-d 倒数第二层，不是分类 logits
        out.append(torch.flatten(f, 1).half().cpu().numpy())   # float16：本机主机内存只有 ~1GB
    return np.concatenate(out, 0)


def main():
    # 预检：这台机器主机内存很紧，先检查再启动，避免跑到一半 ArrayMemoryError
    try:
        import psutil
        av = psutil.virtual_memory().available / 1e9
        print('host RAM available: %.2f GB' % av, flush=True)
        if av < 0.35:
            print('!! 需要至少 0.35GB 主机内存（当前 %.2f GB）。请先关闭夸克客户端/多余浏览器/WSL 再跑。' % av)
            return
    except Exception:
        pass
    torch.set_num_threads(2)          # 24 线程在 15.6GB 机器上会放大内存占用
    nfolds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', dev, 'epochs', epochs, 'folds', nfolds, flush=True)
    res = {a: [] for a in ['V', 'K', 'Kp']}
    t_all = time.time()
    for p in PLIST[:nfolds]:
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(set(y[tr].tolist())) < C:
            continue
        t0 = time.time()
        print('-- fold %s (train %d / test %d)' % (p, len(tr), len(te)), flush=True)
        net = finetune(tr, epochs, dev)
        Ftr = extract(net, tr, dev); Fte = extract(net, te, dev)
        # 存盘：把"抽特征"与"PCA+评估"解耦，内存抖动时只需重跑便宜的那段
        np.savez(OUTF / ('ft_%s.npz' % p.replace('/', '_')), Ftr=Ftr, Fte=Fte, tr=tr, te=te)
        del net
        import gc
        gc.collect(); torch.cuda.empty_cache()
        if os.environ.get('FEAT_ONLY'):
            print('   fold %s: features saved (%.0f s)' % (p, time.time() - t0), flush=True)
            continue
        pca = PCA(n_components=128, random_state=0).fit(Ftr)
        Xtr = pca.transform(Ftr); Xte = pca.transform(Fte)
        sd = Xtr.std(0) + 1e-9
        Xtr = Xtr / sd; Xte = Xte / sd
        def kstd(Kd, i1, i2):
            A, Bm = Kd[i1][:, LEGAL], Kd[i2][:, LEGAL]
            mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
            return (A - mu) / s, (Bm - mu) / s
        A, Bm = kstd(K, tr, te)
        Ap, Bp = kstd(Kp, tr, te)
        best_a, best = 1.0, -1.0
        for a in ALPHAS:
            sc = []
            for q in sorted(set(ports[tr].tolist()))[:3]:
                itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
                if len(ite) < 5 or len(set(y[itr].tolist())) < C:
                    continue
                ai, bi = kstd(K, itr, ite)
                sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(
                    np.c_[Xtr[np.isin(tr, itr)], ai], y[itr]).predict(np.c_[Xtr[np.isin(tr, ite)], bi])))
            if sc and float(np.mean(sc)) > best:
                best, best_a = float(np.mean(sc)), a
        pv = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(Xtr, y[tr]).predict(Xte)
        pk = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(
            np.c_[Xtr, A], y[tr]).predict(np.c_[Xte, Bm])
        pkp = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(
            np.c_[Xtr, Ap], y[tr]).predict(np.c_[Xte, Bp])
        res['V'].append(ba(y[te], pv)); res['K'].append(ba(y[te], pk)); res['Kp'].append(ba(y[te], pkp))
        print('   fold %s: V %.4f  K_raw %.4f (%+.4f)  K_pct %.4f (%+.4f)   (%.0f s)' % (
            p, res['V'][-1], res['K'][-1], res['K'][-1] - res['V'][-1],
            res['Kp'][-1], res['Kp'][-1] - res['V'][-1], time.time() - t0), flush=True)
    print('\n=== fine-tuned baseline (%d folds) ===' % len(res['V']), flush=True)
    mv, mk, mp = (float(np.mean(res[a])) for a in ['V', 'K', 'Kp'])
    print('  V %.4f  K_raw %.4f (Δ %+.4f)  K_pct %.4f (Δ %+.4f)' % (mv, mk, (mk - mv) * 100, mp, (mp - mv) * 100), flush=True)
    import numpy as _np
    dv = (_np.array(res['K']) - _np.array(res['V'])) * 100; dp = (_np.array(res['Kp']) - _np.array(res['V'])) * 100
    print('  逐折: K_raw 正 %d/%d   K_pct 正 %d/%d' % ((dv > 0).sum(), len(dv), (dp > 0).sum(), len(dp)), flush=True)
    print('  总耗时 %.0f s（每折约 %.0f s → 全 24 折约 %.1f 小时）' % (
        time.time() - t_all, (time.time() - t_all) / max(1, len(res['V'])),
        (time.time() - t_all) / max(1, len(res['V'])) * 24 / 3600), flush=True)
    json.dump(res, open(OUTF / 'finetune_folds.json', 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()

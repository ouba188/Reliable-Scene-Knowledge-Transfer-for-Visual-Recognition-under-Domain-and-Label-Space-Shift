"""e104: evaluate the repaired dataset (A2 tier) AND the AIS length as a closed-set feature.

Two questions in one run, because they share the same folds and the same features:
  1. does the AIS-matched label tier fix the base accuracy? the old build (mixed tier) scored 0.215 where the older
     AIS-matched pool scored ~0.44 -- this dataset uses the same tier as that pool, so the base should recover.
  2. does the reliable per-vessel AIS length (97 % coverage, label-free, physical) help the known classes --
     especially the ones the visual model fails (tug, offshore, product-chemical)?
Arms per fold: visual | visual + AIS length | visual + knowledge (local context + facility proximity).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244_q'
W = ROOT / 'weights/resnet50_sentinel1_all_moco.pth'
OUT = ROOT / 'features_244q'
OUT.mkdir(parents=True, exist_ok=True)
CTX = ROOT / 'features_244/localctx_244.npy'
FAC = ROOT / 'features_244/facility_dist_244.npy'
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
rng = np.random.default_rng(0)
SUBS = 20000
SIZE, BATCH = 224, 128
MODE = sys.argv[1] if len(sys.argv) > 1 else 'both'

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
print('dataset244_q chips %d' % len(idx), flush=True)

if MODE in ('extract', 'both'):
    import torch
    from torch import nn
    from torchvision.models import resnet50
    MEAN = np.array([-12.59, -20.26], np.float32); STD = np.array([5.26, 5.91], np.float32)

    def build():
        n = resnet50(weights=None)
        n.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        n.load_state_dict(torch.load(W, map_location='cpu', weights_only=True), strict=False)
        n.fc = nn.Identity(); n.eval()
        return n

    MM = {}

    def mm(pj, pol):
        k = (pj, pol)
        if k not in MM:
            MM[k] = np.load(DS / ('%s_%s.npy' % (pj.replace(' ', '_'), pol)), mmap_mode='r')
        return MM[k]

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build().to(dev).half()
    feats = np.zeros((len(idx), 2048), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            ch = idx[i:i + BATCH]
            xs = []
            for r in ch:
                try:
                    vv = mm(r['port'], 'VV')[int(r['row'])].astype(np.float32)
                    vh = mm(r['port'], 'VH')[int(r['row'])].astype(np.float32)
                    db = 20.0 * np.log10((np.stack([vv, vh], 0) + 0.5) / 255.0 + 1e-6)
                    for c in range(2):
                        db[c] = (db[c] - db[c].mean()) / (db[c].std() + 1e-6) * STD[c] + MEAN[c]
                    xs.append(db)
                except Exception:
                    xs.append(np.zeros((2, SIZE, SIZE), np.float32))
            t = torch.from_numpy(np.stack(xs)).to(dev).half()
            feats[i:i + len(ch)] = net(t).float().cpu().numpy().reshape(len(ch), -1).astype(np.float16)
            if i % (BATCH * 20) == 0 or i + BATCH >= len(idx):
                np.save(OUT / 'resnet50_s1b_244q.float16.npy', feats)
                print('  feat %d/%d %.0fs' % (i + len(ch), len(idx), time.time() - t0), flush=True)
    np.save(OUT / 'resnet50_s1b_244q.float16.npy', feats)
    print('特征完成', feats.shape, flush=True)
if MODE == 'extract':
    sys.exit(0)

X = np.load(OUT / 'resnet50_s1b_244q.float16.npy').astype(np.float32)
AIS = np.load(AISQ, mmap_mode='r')
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
have = np.isfinite(AIS[:, 0])
print('AIS 长度覆盖 %d/%d' % (int(have.sum()), len(idx)), flush=True)


def ba(yy, pred, cc=None):
    rngc = cc if cc is not None else range(8)
    rs = [float((pred[yy == c] == c).mean()) for c in rngc if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


acc = {'visual': [], 'visual+aislen': [], 'hard:visual': [], 'hard:+aislen': []}
for p in sorted(set(ports[known])):
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where((ports == p) & have)[0]
    if len(tr) < 300 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
    am = aq[trs].mean(0, keepdims=True); asd = aq[trs].std(0, keepdims=True) + 1e-6
    Atr = (aq[trs] - am) / asd; Ate = (aq[te] - am) / asd
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    ra = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, Atr], y[trs])
    pv = rv.predict(Zte); pa = ra.predict(np.c_[Zte, Ate])
    tk = known[te]
    acc['visual'].append(ba(y[te][tk], pv[tk])); acc['visual+aislen'].append(ba(y[te][tk], pa[tk]))
    hard = np.isin(y[te], [6, 7, 3])
    m = tk & hard
    if m.sum() >= 6:
        acc['hard:visual'].append(ba(y[te][m], pv[m], [c for c in (6, 7, 3) if (y[te][m] == c).any()]))
        acc['hard:+aislen'].append(ba(y[te][m], pa[m], [c for c in (6, 7, 3) if (y[te][m] == c).any()]))
    print('%-16s 已知 %5d | BA 视觉 %.3f +AIS长度 %.3f' % (
        p, int(tk.sum()), acc['visual'][-1], acc['visual+aislen'][-1]), flush=True)

print('')
print('已知类 BA: 视觉 %.4f | +AIS长度 %.4f   (旧数据集同一基线 0.215；同档老池子 ~0.44)' % (
    np.nanmean(acc['visual']), np.nanmean(acc['visual+aislen'])))
print('难类(拖轮/海工/化学品) 逐港平均: 视觉 %.4f | +AIS长度 %.4f' % (
    np.nanmean(acc['hard:visual']), np.nanmean(acc['hard:+aislen'])))

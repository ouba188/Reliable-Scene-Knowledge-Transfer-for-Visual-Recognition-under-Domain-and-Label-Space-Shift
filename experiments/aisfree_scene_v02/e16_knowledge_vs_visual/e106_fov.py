"""e106: is the 0.19-vs-0.44 gap a FIELD-OF-VIEW effect?

The older 8-class pool cropped a 64-pixel source window (crop_side_vv = 64 in its manifest) and wrote it out at 128,
i.e. a 640 m field of view at 10 m/pixel, zoomed. This rebuilt dataset crops 224 native pixels, i.e. a 2240 m window
-- 3.5x more sea, berth and neighbour clutter around the same ship. Ship TYPE is a fine-grained question, so the
extra context can easily cost accuracy. Test with no re-cropping: take the centre 64 pixels of each stored 224 crop,
upscale to the encoder's input size, re-extract, and compare on the same folds.
Variants: centre64 (the old FOV, 640 m), centre112 (1120 m), and the full 224 (the current baseline).
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
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
SIZE, BATCH = 224, 128
rng = np.random.default_rng(0)
SUBS = 20000
MODE = sys.argv[1] if len(sys.argv) > 1 else 'both'

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
print('chips %d' % len(idx), flush=True)
VARIANTS = {'c64': 64, 'c112': 112, 'full224': 224}

import torch
from PIL import Image
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


def center_crop_resize(arr, keep):
    if keep >= arr.shape[0]:
        return arr.astype(np.float32)
    o = (arr.shape[0] - keep) // 2
    sub = arr[o:o + keep, o:o + keep]
    im = Image.fromarray(sub).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im, np.float32)


if MODE in ('extract', 'both'):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build().to(dev).half()
    for vname, keep in VARIANTS.items():
        dest = OUT / ('resnet50_%s.float16.npy' % vname)
        if dest.is_file():
            print('%s 已存在，跳过' % vname, flush=True)
            continue
        feats = np.zeros((len(idx), 2048), np.float16)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(idx), BATCH):
                ch = idx[i:i + BATCH]
                xs = []
                for r in ch:
                    try:
                        vv = center_crop_resize(mm(r['port'], 'VV')[int(r['row'])], keep)
                        vh = center_crop_resize(mm(r['port'], 'VH')[int(r['row'])], keep)
                        db = 20.0 * np.log10((np.stack([vv, vh], 0) + 0.5) / 255.0 + 1e-6)
                        for c in range(2):
                            db[c] = (db[c] - db[c].mean()) / (db[c].std() + 1e-6) * STD[c] + MEAN[c]
                        xs.append(db)
                    except Exception:
                        xs.append(np.zeros((2, SIZE, SIZE), np.float32))
                t = torch.from_numpy(np.stack(xs)).to(dev).half()
                feats[i:i + len(ch)] = net(t).float().cpu().numpy().reshape(len(ch), -1).astype(np.float16)
                if i % (BATCH * 40) == 0 or i + BATCH >= len(idx):
                    print('  %s %d/%d %.0fs' % (vname, i + len(ch), len(idx), time.time() - t0), flush=True)
        np.save(dest, feats)
        print('%s DONE %s (%.0fs)' % (vname, feats.shape, time.time() - t0), flush=True)
if MODE == 'extract':
    sys.exit(0)

cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
AIS = np.load(AISQ, mmap_mode='r')
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have = np.isfinite(AIS[:, 0])


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


print('')
print('%-10s %10s %12s' % ('variant', 'FOV(m)', '已知类BA'))
for vname, keep in VARIANTS.items():
    f = OUT / ('resnet50_%s.float16.npy' % vname)
    if not f.is_file():
        print('%-10s 缺特征' % vname); continue
    X = np.load(f).astype(np.float32)
    accs = []
    for p in sorted(set(ports[known])):
        tr = np.where(known & (ports != p) & have)[0]
        te = np.where((ports == p) & have)[0]
        if len(tr) < 300 or len(te) < 20:
            continue
        trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
        mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
        rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mu) / sd, y[trs])
        pv = rv.predict((X[te] - mu) / sd)
        tk = known[te]
        accs.append(ba(y[te][tk], pv[tk]))
    if accs:
        print('%-10s %10d %12.4f   (%d 港)' % (vname, keep * 10, float(np.mean(accs)), len(accs)))
print('')
print('参考：老池子 64px 源窗（640 m FOV）在同档标签下 ~0.44')

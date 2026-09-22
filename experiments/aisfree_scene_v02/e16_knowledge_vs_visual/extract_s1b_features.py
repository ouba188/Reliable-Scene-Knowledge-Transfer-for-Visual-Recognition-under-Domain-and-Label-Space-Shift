"""Extract ResNet50 features with SAR-pretrained weights AND the CORRECT input preprocessing.

torchgeo `resnet50_sentinel1_all_moco` expects Sentinel-1 **dB** input:
    transforms = Resize(256) -> CenterCrop(224) -> Normalize(mean=[-12.59,-20.26], std=[5.26,5.91])
Our chips are 8-bit = clip((DN16 - lo)/(hi - lo)*255) (2%/98% percentiles, linear amplitude), so
dB = 20*log10(DN) + const. With lo/hi (per-product) unavailable, we restore the log structure and
match the expected dB statistics per channel:

    dn_rel = (x + 0.5) / 255                      # linear amplitude, relative to the clip window
    db     = 20*log10(dn_rel)                     # log domain up to a constant
    z      = (db - mean(db)) / std(db) * std_s1 + mean_s1

ponytail: constant offset absorbed by the statistics match; if the per-product lo/hi ever reappear,
swap dn_rel for the true (dn16-lo)/(hi-lo).
"""
import csv, json, os, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from PIL import Image

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
W = Path(r'E:/临时会话/visual_reliable_baseline/weights/resnet50_sentinel1_all_moco.pth')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/features_s1b')
OUT.mkdir(parents=True, exist_ok=True)
MEAN_S1 = np.array([-12.59, -20.26], np.float32)
STD_S1 = np.array([5.26, 5.91], np.float32)
BATCH = 96
SIZE = 224


def build():
    net = resnet50(weights=None)
    net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    sd = torch.load(W, map_location='cpu', weights_only=True)
    net.load_state_dict(sd, strict=False)
    net.fc = nn.Identity()
    net.eval()
    return net


def load_pair(row):
    a = Image.open(row['image_relpath_vv']).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    b = Image.open(row['image_relpath_vh']).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x = np.stack([np.asarray(a, np.float32), np.asarray(b, np.float32)], 0)
    dn = (x + 0.5) / 255.0
    db = 20.0 * np.log10(dn + 1e-6)
    for c in range(2):
        m, s = db[c].mean(), db[c].std() + 1e-6
        db[c] = (db[c] - m) / s * STD_S1[c] + MEAN_S1[c]
    return db


def main():
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
    n = len(rows)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build().to(dev).half()
    feats = np.zeros((n, 2048), np.float16)
    prog = OUT / 'feature_progress.json'
    start = 0
    if prog.exists():
        d = json.loads(prog.read_text())
        if d.get('total_rows') == n:
            start = int(d.get('completed_rows', 0))
            f = OUT / 'resnet50_s1b.float16.npy'
            if f.exists():
                feats[:start] = np.load(f, mmap_mode='r')[:start]
    print('start', start, '/', n, flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i in range(start, n, BATCH):
            ch = rows[i:i + BATCH]
            xs = []
            for r in ch:
                try:
                    xs.append(load_pair(r))
                except Exception:
                    xs.append(np.zeros((2, SIZE, SIZE), np.float32))
            t = torch.from_numpy(np.stack(xs)).to(dev).half()
            o = net(t).float().cpu().numpy().reshape(len(ch), -1)
            feats[i:i + len(ch)] = o.astype(np.float16)
            if i % (BATCH * 10) == 0 or i + BATCH >= n:
                np.save(OUT / 'resnet50_s1b.float16.npy', feats)
                prog.write_text(json.dumps({'total_rows': n, 'completed_rows': i + len(ch)}))
                el = time.time() - t0
                print('%d/%d %.1fs %.1f it/s' % (i + len(ch), n, el, (i + len(ch) - start) / max(1e-9, el)), flush=True)
    np.save(OUT / 'resnet50_s1b.float16.npy', feats)
    prog.write_text(json.dumps({'total_rows': n, 'completed_rows': n}))
    print('DONE', feats.shape, flush=True)


if __name__ == '__main__':
    main()

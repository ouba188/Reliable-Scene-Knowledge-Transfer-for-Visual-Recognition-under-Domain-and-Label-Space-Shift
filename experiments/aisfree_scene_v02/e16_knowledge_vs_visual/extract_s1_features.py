"""Extract ResNet50 features with SAR-pretrained weights (torchgeo resnet50_sentinel1_all_moco).

Only the WEIGHTS differ from the current pipeline: same VV/VH 2-channel input, same 128 bilinear,
same (x/255-0.5)/0.25 normalization, same layer4 pooled2x2 ->> spatial mean. That isolates the
'ImageNet vs SAR-pretrained encoder' effect.

Resumable: writes feature_progress.json + the npy incrementally.
"""
import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from PIL import Image

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
W = Path(r'E:/临时会话/visual_reliable_baseline/weights/resnet50_sentinel1_all_moco.pth')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/features_s1')
OUT.mkdir(parents=True, exist_ok=True)
BATCH = 128


def build():
    net = resnet50(weights=None)
    # the SAR checkpoint has a native 2-channel conv1 (VV+VH) -> swap it in before loading
    net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # safe load: this checkpoint is a plain state_dict of tensors (torchgeo HF release), so
    # weights_only=True suffices and avoids unpickling arbitrary objects.
    sd = torch.load(W, map_location='cpu', weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print('missing:', len(missing), 'unexpected:', len(unexpected), flush=True)
    for m in missing[:5]:
        print('  missing key:', m)
    net.fc = nn.Identity()
    net.eval()
    return net


def load_pair(row):
    a = Image.open(row['image_relpath_vv']).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
    b = Image.open(row['image_relpath_vh']).convert('L').resize((128, 128), Image.Resampling.BILINEAR)
    x = np.stack([np.asarray(a, np.float32), np.asarray(b, np.float32)], 0)
    return (x / 255.0 - 0.5) / 0.25


def main():
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
    n = len(rows)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build().to(dev).half() if dev.type == 'cuda' else build().to(dev)
    feats = np.zeros((n, 2048), np.float16)
    prog = OUT / 'feature_progress.json'
    start = 0
    if prog.exists():
        d = json.loads(prog.read_text())
        if d.get('total_rows') == n:
            start = int(d.get('completed_rows', 0))
            f = OUT / 'resnet50_s1.float16.npy'
            if f.exists():
                feats[:start] = np.load(f, mmap_mode='r')[:start]
    print('start at', start, 'of', n, flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i in range(start, n, BATCH):
            chunk = rows[i:i + BATCH]
            xs = []
            for r in chunk:
                try:
                    xs.append(load_pair(r))
                except Exception:
                    xs.append(np.zeros((2, 128, 128), np.float32))
            t = torch.from_numpy(np.stack(xs)).to(dev)
            if dev.type == 'cuda':
                t = t.half()
            f = net(t)                      # (B,2048,1,1) after layer4+avgpool by resnet50
            f = f.float().cpu().numpy().reshape(len(chunk), -1)
            feats[i:i + len(chunk)] = f.astype(np.float16)
            if i % (BATCH * 10) == 0 or i + BATCH >= n:
                np.save(OUT / 'resnet50_s1.float16.npy', feats)
                prog.write_text(json.dumps({'total_rows': n, 'completed_rows': i + len(chunk)}))
                el = time.time() - t0
                print('%d/%d  %.1f s  %.1f it/s' % (i + len(chunk), n, el, (i + len(chunk) - start) / max(1e-9, el)), flush=True)
    np.save(OUT / 'resnet50_s1.float16.npy', feats)
    prog.write_text(json.dumps({'total_rows': n, 'completed_rows': n}))
    print('DONE ->', OUT / 'resnet50_s1.float16.npy', feats.shape, flush=True)


if __name__ == '__main__':
    main()

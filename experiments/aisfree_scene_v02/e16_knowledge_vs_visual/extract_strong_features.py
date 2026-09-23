"""extract_dinov2_features.py -- stronger visual encoders for the 38,091 chips.

Usage: python extract_dinov2_features.py <timm_name> <tag>
e.g.   python extract_dinov2_features.py vit_small_patch14_dinov2.lvd142m dinov2s

2-channel (VV, VH) input via timm's automatic patch-embed adaptation; 224x224 (patch-14 models need
a multiple of 14); ImageNet-style normalisation from the model's own data config. Resumable.
"""
import csv, json, sys, time, os
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/features_strong')
OUT.mkdir(parents=True, exist_ok=True)
NAME = sys.argv[1] if len(sys.argv) > 1 else 'vit_small_patch14_dinov2.lvd142m'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'dinov2s'
SIZE = 224
CHUNK = 2000
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def main():
    rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
    n = len(rows)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('model', NAME, 'device', dev, 'n', n, flush=True)
    m = timm.create_model(NAME, pretrained=True, in_chans=2, num_classes=0,
                          dynamic_img_size=True).to(dev).eval()   # 位置编码可插值 → 224 可用
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg['mean'])[:2].view(1, 2, 1, 1).to(dev)
    std = torch.tensor(cfg['std'])[:2].view(1, 2, 1, 1).to(dev)
    D = m.num_features
    f = OUT / ('feat_%s.float16.npy' % TAG)
    prog = OUT / ('progress_%s.json' % TAG)
    start = 0
    if f.exists() and prog.exists():
        d = json.loads(prog.read_text())
        if d.get('total') == n:
            start = int(d.get('done', 0))
    if f.exists() and start == 0:
        out = np.lib.format.open_memmap(f, mode='r+')
    else:
        out = np.lib.format.open_memmap(f, mode='w+' if start == 0 else 'r+', dtype=np.float16, shape=(n, D))
    print('dim', D, 'start', start, flush=True)
    t0 = time.time(); miss = 0
    with torch.no_grad():
        for i in range(start, n, CHUNK):
            batch = rows[i:i + CHUNK]
            xs = []
            for r in batch:
                try:
                    a = Image.open(r['image_relpath_vv']).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                    b = Image.open(r['image_relpath_vh']).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                    xs.append(np.stack([np.asarray(a, np.float32), np.asarray(b, np.float32)], 0))
                except Exception:
                    miss += 1
                    xs.append(np.zeros((2, SIZE, SIZE), np.float32))
            x = torch.from_numpy(np.stack(xs)).to(dev)
            x = (x / 255.0 - mean) / std
            h = m(x).float()
            out[i:i + len(x)] = h.half().cpu().numpy()
            out.flush()
            prog.write_text(json.dumps({'total': n, 'done': min(i + CHUNK, n), 'missing': miss, 'dim': D}))
            el = time.time() - t0
            print('%d/%d %.0fs %.1f it/s miss=%d' % (min(i + CHUNK, n), n, el, (min(i + CHUNK, n) - start) / max(1e-9, el), miss), flush=True)
    print('DONE', f, 'missing', miss, flush=True)


if __name__ == '__main__':
    main()

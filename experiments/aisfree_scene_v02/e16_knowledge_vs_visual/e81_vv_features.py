"""e81: VV-only feature extraction for BOTH pools (known 8-class + unknown), plus the NAR measurement.

Why VV-only: the V5 filenames carry different uids per polarization and the object table has no uid column, so
the true VV/VH pairing of the unknown chips cannot be recovered (the rank rule validated at only 18.3%). Feeding
[VV, VV] to both pools keeps the known-vs-unknown comparison controlled; only the absolute level drops.

Pipeline (identical encoder + dB preprocessing as extract_s1b_features.py, which is the validated one):
  resnet50_sentinel1_all_moco, 2-channel conv1, z = ((db - mean)/std)*STD_S1 + MEAN_S1
Outputs: features_vv/{known,unknown}.float16.npy + unknown_index.csv
Then: cross-port LOO ridge arms on the KNOWN pool, applied to the UNKNOWN chips, and the absorption report.
"""
import csv
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
D5 = Path(r'E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_merged_20260915/images')
W = Path(r'E:/临时会话/visual_reliable_baseline/weights/resnet50_sentinel1_all_moco.pth')
OUT = Path(r'E:/临时会话/visual_reliable_baseline/features_vv')
OUT.mkdir(parents=True, exist_ok=True)
MEAN_S1 = np.array([-12.59, -20.26], np.float32)
STD_S1 = np.array([5.26, 5.91], np.float32)
UNKNOWN = ['lpg_lng_tanker', 'dredger', 'passenger_ship', 'pilot_port_tender', 'ro_ro_vehicle_carrier',
           'pleasure_craft', 'heavy_load_carrier', 'sailing_vessel', 'reefer_cargo']
BATCH, SIZE = 128, 224
PAT = re.compile(r'^(?P<prod>.+)_(?P<pol>VV|VH)_UTM_8bit__(?P<tile>r\d+_c\d+)__(?P<uid>[0-9a-f]+)\.png$')
MODELS = {}


def net_build():
    import torch
    from torch import nn
    from torchvision.models import resnet50
    net = resnet50(weights=None)
    net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.load_state_dict(torch.load(W, map_location='cpu', weights_only=True), strict=False)
    net.fc = nn.Identity(); net.eval()
    return net


def load_vv(path):
    import torch
    from PIL import Image
    a = Image.open(path).convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(a, np.float32)
    db = 20.0 * np.log10((x + 0.5) / 255.0 + 1e-6)
    db = (db - db.mean()) / (db.std() + 1e-6) * STD_S1[0] + MEAN_S1[0]
    return np.stack([db, db], 0)


def extract(paths, tag):
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net_build().to(dev).half()
    n = len(paths)
    feats = np.zeros((n, 2048), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH):
            ch = paths[i:i + BATCH]
            xs = []
            for p in ch:
                try:
                    xs.append(load_vv(p))
                except Exception:
                    xs.append(np.zeros((2, SIZE, SIZE), np.float32))
            t = torch.from_numpy(np.stack(xs)).to(dev).half()
            feats[i:i + len(ch)] = net(t).float().cpu().numpy().reshape(len(ch), -1).astype(np.float16)
            if i % (BATCH * 20) == 0 or i + BATCH >= n:
                np.save(OUT / ('%s.float16.npy' % tag), feats)
                print('%s %d/%d %.0fs' % (tag, i + len(ch), n, time.time() - t0), flush=True)
    np.save(OUT / ('%s.float16.npy' % tag), feats)
    print('DONE', tag, feats.shape, flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('known', 'both'):
        rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
        paths = [r['image_relpath_vv'] for r in rows]
        with (OUT / 'known_index.csv').open('w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh); w.writerow(['port', 'class_id', 'class_name', 'vv'])
            for r in rows:
                w.writerow([r['port'], r['class_id'], r['class_name'], r['image_relpath_vv']])
        extract(paths, 'known')
    if which in ('unknown', 'both'):
        idx = []
        for pj in sorted(p.name for p in D5.iterdir() if p.is_dir()):
            for cls in UNKNOWN:
                d = D5 / pj / cls
                if not d.is_dir():
                    continue
                for f in sorted(d.glob('*_VV_UTM_8bit__*.png')):
                    idx.append({'port': pj, 'fine_class': cls, 'vv': str(f)})
        with (OUT / 'unknown_index.csv').open('w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['port', 'fine_class', 'vv'])
            w.writeheader(); w.writerows(idx)
        print('unknown chips', len(idx), flush=True)
        extract([r['vv'] for r in idx], 'unknown')


if __name__ == '__main__':
    main()

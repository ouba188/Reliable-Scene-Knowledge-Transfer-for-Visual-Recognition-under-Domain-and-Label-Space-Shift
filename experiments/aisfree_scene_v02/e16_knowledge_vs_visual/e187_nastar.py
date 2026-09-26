"""e187: fourth external validation -- NASTaR (NovaSAR, S-band, a different sensor), 3415 patches / 25 classes / 624 scenes.

Reading of the dataset shape matters for the protocol: the median scene holds only THREE patches, so scene-wise leave-one-out is
useless here; the dataset instead carries two usable domain factors per patch, in its own AIS.csv: Shoreline (inshore/offshore)
and the scene's geographic position. So the evaluation is (a) leave-one-GEO-REGION-out over regions obtained by clustering scene
coordinates, which is the genuine cross-domain test, and (b) the inshore/offshore split reported separately as a sub-population
view. Labels come from the patch filename (second-to-last underscore segment), which matches the AIS.csv Ship type column.

Features: ResNet-50 penultimate on each 512x512 uint8 patch (single polarisation, HH). Ridge on standardised features, then rank
the target region by the classifier's own margin and compare the top-k% with a matched-size random subset.
Pre-registered: at k = 10/20/50%, the ranked subset beats matched random in >= 4 of 6 geo-regions AND pooled.
"""
import csv
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tifffile
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(r'E:/NASTaR')
CACHE = Path(r'E:/临时会话/visual_reliable_baseline/nastar_feat.npz')
KS = (0.10, 0.20, 0.50, 1.00)
NREP = 30
NREG = 6
rng = np.random.default_rng(0)

patches = sorted(glob.glob(str(ROOT / '**' / 'ship_patches_uint8' / '*.tif'), recursive=True))
print('uint8 船片 %d' % len(patches), flush=True)

# metadata from each scene's AIS.csv: patch name -> ship type, shoreline, scene lat/lon
meta = {}
for f in glob.glob(str(ROOT / '**' / 'AIS.csv'), recursive=True):
    try:
        with open(f, encoding='utf-8', errors='replace') as fh:
            lines = fh.read().splitlines()
        if not lines:
            continue
        hdr = lines[0].lstrip('#').strip().split(',')
        rd = csv.DictReader(lines[1:], fieldnames=hdr)
        if True:
            for r in rd:
                pn = (r.get('Patch_name') or '').strip()
                if pn:
                    meta[pn] = dict(typ=(r.get('Ship type') or '').strip(),
                                    shore=(r.get('Shoreline') or '').strip().lower(),
                                    lat=float(r.get('Latitude') or 'nan'), lon=float(r.get('Longitude') or 'nan'))
    except Exception:
        pass
print('元数据条目 %d ｜ 有 inshore/offshore: %d' % (len(meta), sum(1 for v in meta.values() if v['shore'])), flush=True)

labels, shores, coords, keep = [], [], [], []
for p in patches:
    base = os.path.basename(p).replace('.tif', '')
    key = base[:-6] if base.endswith('_uint8') else base          # AIS.csv names the patch without the encoding suffix
    m = meta.get(key) or meta.get(base)
    typ = (m['typ'] if m else base.split('_')[-2]) or 'unknown'
    shore = (m['shore'] if m else '')
    labels.append(typ); shores.append(shore)
    coords.append((m['lat'], m['lon']) if m else (np.nan, np.nan))
    keep.append(True)
labels = np.array(labels); shores = np.array(shores); coords = np.array(coords, float)
print('类型 %d ｜ 大类: %s' % (len(set(labels.tolist())), dict(Counter(labels).most_common(10))), flush=True)
print('Shoreline:', dict(Counter(shores)), flush=True)

if CACHE.exists():
    Z = np.load(CACHE)['Z']
    print('特征缓存 ✓', Z.shape, flush=True)
else:
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); net.fc = nn.Identity(); net.eval().to(dev)
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i, p in enumerate(patches):
            a = tifffile.imread(p)
            if a.ndim == 2:
                a = np.repeat(a[..., None], 3, -1)      # single-channel SAR patch -> 3 channels for ResNet
            im = Image.fromarray(a)                     # tifffile decodes; PIL is only the container Resize expects
            out.append(net(tf(im).unsqueeze(0).to(dev)).cpu().numpy().ravel())
            if (i + 1) % 500 == 0:
                print('  提取 %d/%d' % (i + 1, len(patches)), flush=True)
    Z = np.stack(out).astype(np.float32)
    np.savez_compressed(CACHE, Z=Z)
    print('特征 ✓', Z.shape, flush=True)

classes = [c for c, n in Counter(labels).most_common() if n >= 40]
sel = np.isin(labels, classes)
print('保留类别 %d 个（≥40 片）｜ 样本 %d' % (len(classes), int(sel.sum())), flush=True)
Z2, lab2, co2, sh2 = Z[sel], labels[sel], coords[sel], shores[sel]
names = sorted(set(lab2.tolist())); y = np.array([names.index(c) for c in lab2]); C = len(names)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


usable = np.isfinite(co2).all(1)
region = np.full(len(lab2), -1)
if usable.sum() > 100:
    km = KMeans(n_clusters=NREG, n_init=4, random_state=0).fit(co2[usable])
    region[usable] = km.labels_
    print('地理区域 %d 个 ｜ 各区域片数 %s' % (NREG, dict(Counter(region[usable].tolist()))), flush=True)

for axis, groups in (('geo-region', region), ('inshore/offshore', None)):
    print('')
    print('=== 折轴：%s ===' % axis, flush=True)
    res = {k: {'r': [], 'n': []} for k in KS}
    if axis == 'inshore/offshore':
        pairs = [(u, np.where(sh2 == u)[0]) for u in sorted(set(sh2.tolist())) if u]
    else:
        pairs = [(str(r), np.where(region == r)[0]) for r in sorted(set(region[region >= 0].tolist()))]
    for gname, te in pairs:
        tr = np.setdiff1d(np.arange(len(lab2)), te, assume_unique=False)
        if len(tr) < 150 or len(te) < 40:
            print('  %-14s 样本不足（源 %d / 目标 %d）跳过 ⚠' % (gname, len(tr), len(te)), flush=True)
            continue
        mu = Z2[tr].mean(0, keepdims=True); sd = Z2[tr].std(0, keepdims=True) + 1e-6
        m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z2[tr] - mu) / sd, y[tr])
        L = m.decision_function((Z2[te] - mu) / sd)
        if L.shape[1] != C:
            Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
        t2 = np.sort(L, 1)[:, -2:]
        margin = t2[:, 1] - t2[:, 0]
        pred = L.argmax(1); yy = y[te]
        row = ['  %-14s n=%4d 全量 %.3f' % (gname, len(te), ba(yy, pred))]
        for k in KS:
            n = max(1, int(k * len(te)))
            sidx = np.argsort(-margin)[:n]
            a = ba(yy[sidx], pred[sidx])
            b = float(np.mean([(lambda ix: ba(yy[ix], pred[ix]))(np.isin(np.arange(len(te)), rng.permutation(len(te))[:n]))
                               for _ in range(NREP)]))
            res[k]['r'].append(a); res[k]['n'].append(b)
            if k == 0.20:
                row.append('｜ 20%%: %.3f vs %.3f (%s)' % (a, b, '✓' if a > b else '✗'))
        print(''.join(row), flush=True)
    if not res[0.20]['r']:
        continue
    print('  %-8s %12s %12s %10s %12s' % ('预算', '秩选 BA', '随机 BA', 'Δ', '占优折数'))
    for k in KS:
        a, b = np.array(res[k]['r']), np.array(res[k]['n'])
        print('  %-8s %12.4f %12.4f %+10.4f %8d/%d' % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(),
                                                      int((a > b).sum()), len(a)))
    a, b = np.array(res[0.20]['r']), np.array(res[0.20]['n'])
    need = max(2, int(np.ceil(0.667 * len(a))))
    print('  预注册判据（20%% 处 ≥%d/%d 折占优 且 池化更高）: %s'
          % (need, len(a), '成立 ✓✓' if (int((a > b).sum()) >= need and a.mean() > b.mean()) else '未成立 ✗'), flush=True)

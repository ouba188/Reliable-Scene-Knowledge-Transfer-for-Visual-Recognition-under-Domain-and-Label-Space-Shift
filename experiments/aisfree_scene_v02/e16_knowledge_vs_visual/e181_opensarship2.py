"""e181: third external validation -- OpenSARShip 2.0, on the SAME five folds the local V1 study used.

Assets (local, zero download): opensarship2_derived_manifests.zip carries opensarship2_interference0_shared3_grd_sizeclean.csv
(n=2,469 clean GRD samples, three shared classes tanker 1211 / container 729 / bulk 529) with a PRE-BUILT five-fold cross-port
split (fold_id 0..4 over pub2_region_00..04). Those are the five regions the frozen V1 design held out, so this run shares the
division of labour with V1 and needs no realignment.

Harness identical to e179/e180: ResNet-50 penultimate features on VV|VH, leave-one-region-out by fold_id, rank the target region
by the classifier's own margin, compare the top-k% with a matched-size random subset (30 redraws).
Pre-registered (mirroring V1's own 4/5 rule): at each budget the ranked subset beats the matched random subset in >= 4 of 5 folds
AND pooled. Tiny folds (<30 samples) are reported but excluded from the count.
"""
import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

BASE = Path(r'D:/Documents/project_root/project_data')
ZIP = BASE / 'opensarship2_derived_manifests.zip'
MAN = 'opensarship2_interference0_shared3_grd_sizeclean.csv'
ROOTS = [BASE / 'public_opensarship2', BASE / 'public_opensarship2' / 'opensarship2', BASE / 'OpenSARShip_2']
CACHE = Path(r'E:/临时会话/visual_reliable_baseline/os2_feat.npz')
KS = (0.10, 0.20, 0.50, 1.00)
NREP = 30
rng = np.random.default_rng(0)

with zipfile.ZipFile(ZIP) as f:
    rows = list(csv.DictReader(io.StringIO(f.read(MAN).decode('utf-8-sig', errors='replace'))))
print('样本 %d ｜ 类 %s ｜ fold %s' % (len(rows), dict(Counter(r['label_shared'] for r in rows)),
                                      dict(Counter(r['fold_id'] for r in rows))), flush=True)


def chip(r, pol):
    rel = r['chip_path_%s' % pol] or r['image_path_%s' % pol]
    for root in ROOTS:
        for cand in (root / rel, root / rel.replace('opensarship2/', '', 1), root / 'chips' / pol / Path(rel).name):
            if cand.exists():
                return cand
    return None


p0 = chip(rows[0], 'vv')
print('样例芯片:', p0, flush=True)
if p0 is None:
    raise SystemExit('chip root not found -- check ROOTS')

if CACHE.exists():
    Z = np.load(CACHE)['Z']
    print('特征缓存命中 ✓', Z.shape, flush=True)
else:
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Identity(); net.eval().to(dev)
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            vs = []
            for pol in ('vv', 'vh'):
                p = chip(r, pol) or chip(r, 'vv')
                vs.append(net(tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)).cpu().numpy().ravel())
            feats.append(np.concatenate(vs))
            if (i + 1) % 300 == 0:
                print('  提取 %d/%d' % (i + 1, len(rows)), flush=True)
    Z = np.stack(feats).astype(np.float32)
    np.savez_compressed(CACHE, Z=Z)
    print('特征 ✓ %s' % (Z.shape,), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(int(yy.max()) + 1) if (yy == c).any()]
    return float(np.mean(rs))


names = sorted(set(r['label_shared'] for r in rows))
y = np.array([names.index(r['label_shared']) for r in rows])
fold = np.array([r['fold_id'] for r in rows])
C = len(names)
res = {k: {'r': [], 'n': []} for k in KS}
for f in sorted(set(fold.tolist())):
    tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
    if len(tr) < 100 or len(te) < 30:
        print('  fold %-3s 样本不足（源 %d / 目标 %d）跳过 ⚠' % (f, len(tr), len(te)), flush=True)
        continue
    mu = Z[tr].mean(0, keepdims=True); sd = Z[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z[tr] - mu) / sd, y[tr])
    L = m.decision_function((Z[te] - mu) / sd)
    if L.shape[1] != C:
        Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    pred = L.argmax(1); yy = y[te]
    row = ['  fold %-3s n=%4d 全量 %.3f' % (f, len(te), ba(yy, pred))]
    for k in KS:
        n = max(1, int(k * len(te)))
        seli = np.argsort(-margin)[:n]
        a = ba(yy[seli], pred[seli])
        b = float(np.mean([(lambda idx: ba(yy[idx], pred[idx]))(rng.permutation(len(te))[:n]) for _ in range(NREP)]))
        res[k]['r'].append(a); res[k]['n'].append(b)
        if k == 0.10:
            row.append('｜ 10%%: %.3f vs %.3f %s' % (a, b, '✓' if a > b else '✗'))
    print(''.join(row), flush=True)

print('')
print('%-8s %14s %14s %10s %12s' % ('预算', '秩选 BA', '随机 BA', 'Δ', '占优折数'))
for k in KS:
    a, b = np.array(res[k]['r']), np.array(res[k]['n'])
    if not len(a):
        continue
    print('%-8s %14.4f %14.4f %+10.4f %8d/%d' % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(),
                                                 int((a > b).sum()), len(a)))
for k in (0.10, 0.20, 0.50):
    a, b = np.array(res[k]['r']), np.array(res[k]['n'])
    ok = int((a > b).sum()) >= 4 and a.mean() > b.mean()
    print('预注册判据 %s（≥4/5 折占优 且 池化更高）: %s' % ('%.0f%%' % (100 * k), '成立 ✓✓' if ok else '未成立 ✗'))

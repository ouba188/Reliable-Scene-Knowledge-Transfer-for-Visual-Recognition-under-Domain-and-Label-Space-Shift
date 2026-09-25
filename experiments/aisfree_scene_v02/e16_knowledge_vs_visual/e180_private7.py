"""e180: second external validation -- the private 7-port pool, same method, same controls.

Assets (local): project_data/private_7ports/metadata/private7_manifest_std64_osm.csv -- 1,292 chips over SEVEN ports
(52-488 each), three classes (tanker 803 / container 237 / bulk 252), VV|VH std64 chips present, OSM context columns alongside.

Seven ports means a genuine leave-one-port-out with seven folds and a per-port paired comparison (n=7: report the sign test,
do not lean on a p-value). Same mechanism as e179: rank the target port by the classifier's own margin and compare the top-k%
against a matched-size random subset.
Pre-registered: at each budget the ranked subset must beat the matched random subset in >= 5 of 7 ports AND pooled.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

MAN = Path(r'D:/Documents/project_root/project_data/private_7ports/metadata/private7_manifest_std64_osm.csv')
CACHE = Path(r'E:/临时会话/visual_reliable_baseline/private7_feat.npz')
KS = (0.10, 0.20, 0.50, 1.00)
NREP = 30
rng = np.random.default_rng(0)

rows = list(csv.DictReader(MAN.open(encoding='utf-8-sig')))
ports = np.array([r['port_id'] for r in rows])
lab = np.array([r['label_raw'] for r in rows])
print('样本 %d ｜ 港 %s ｜ 类 %s' % (len(rows), dict(Counter(ports.tolist())), dict(Counter(lab.tolist()))), flush=True)

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
            for col in ('chip_path_vv', 'chip_path_vh'):
                p = r.get(col) or r['chip_path_vv']
                vs.append(net(tf(Image.open(p).convert('RGB')).unsqueeze(0).to(dev)).cpu().numpy().ravel())
            feats.append(np.concatenate(vs))
            if (i + 1) % 200 == 0:
                print('  提取 %d/%d' % (i + 1, len(rows)), flush=True)
    Z = np.stack(feats).astype(np.float32)
    np.savez_compressed(CACHE, Z=Z)
    print('特征 ✓ %s' % (Z.shape,), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(int(yy.max()) + 1) if (yy == c).any()]
    return float(np.mean(rs))


names = sorted(set(lab.tolist()))
y = np.array([names.index(c) for c in lab])
C = len(names)
res = {k: {'r': [], 'n': []} for k in KS}
for p in sorted(set(ports.tolist())):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 60 or len(te) < 30:
        print('  %-12s 样本不足（源 %d / 目标 %d）跳过 ⚠' % (p, len(tr), len(te)), flush=True)
        continue
    mu = Z[tr].mean(0, keepdims=True); sd = Z[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z[tr] - mu) / sd, y[tr])
    L = m.decision_function((Z[te] - mu) / sd)
    if L.shape[1] != C:
        Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    pred = L.argmax(1); yy = y[te]
    row = ['  %-12s n=%4d 全量 %.3f' % (p, len(te), ba(yy, pred))]
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
print('%-8s %14s %14s %10s %12s' % ('预算', '秩选 BA', '随机 BA', 'Δ', '占优港数'))
for k in KS:
    a, b = np.array(res[k]['r']), np.array(res[k]['n'])
    if not len(a):
        continue
    print('%-8s %14.4f %14.4f %+10.4f %8d/%d' % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(),
                                                 int((a > b).sum()), len(a)))
a, b = np.array(res[0.10]['r']), np.array(res[0.10]['n'])
ok10 = int((a > b).sum()) >= 5 and a.mean() > b.mean()
a5, b5 = np.array(res[0.50]['r']), np.array(res[0.50]['n'])
ok50 = int((a5 > b5).sum()) >= 5 and a5.mean() > b5.mean()
print('')
print('预注册判据 10%%（≥5/7 港占优 且 池化更高）: %s' % ('成立 ✓✓' if ok10 else '未成立 ✗'))
print('预注册判据 50%%（同上）: %s' % ('成立 ✓✓' if ok50 else '未成立 ✗'))

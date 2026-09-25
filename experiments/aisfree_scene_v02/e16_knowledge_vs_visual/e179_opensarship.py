"""e179: external validation on OpenSARShip (local, zero download) -- tonight's two-level method, both granularities.

Assets: D:/Documents/Port/public_opensarship/manifests/opensarship_crossport_lopo_3fold_geo_osm_v1.csv, 1,394 chips across three
Chinese ports (Shanghai 191 / Shenzhen 156 / Tianjin 1047), all present locally, with BOTH a fine label (14 classes) and a shared
coarse label (3 classes = the V1 study's 'shared3'), plus OSM/geo context columns.

Method tested (the same shape as tonight's, no knowledge block needed):
  V            ridge on ResNet-50 penultimate features of the VV|VH chip pair
  budget k     rank the target port by the classifier's own margin; report BA on the top k% against a matched-size random subset
  block level  skip the partition here: with three ports the cross-port pairwise separability would be fitted on two ports, which
               is too thin to be meaningful. Recorded as not attempted rather than run and over-read.

Granularities: the coarse 3-class label (all folds feasible) and the fine label restricted to classes with >= 20 samples.
Honest statistics: three ports means a paired test has no power, so per-port values are reported and no p-value is claimed.
Pre-registered: at each budget the ranked subset must beat the matched random subset in >= 2 of 3 ports AND pooled.
"""
import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

MAN = Path(r'D:/Documents/Port/public_opensarship/manifests/opensarship_crossport_lopo_3fold_geo_osm_v1.csv')
CACHE = Path(r'E:/临时会话/visual_reliable_baseline/opensarship_feat.npz')
KS = (0.10, 0.20, 0.50, 1.00)
NREP = 30
rng = np.random.default_rng(0)

rows = list(csv.DictReader(MAN.open(encoding='utf-8-sig')))
ports = np.array([r['port_id'] for r in rows])
fine = np.array([r['label_raw'] for r in rows])
coarse = np.array([r['label_shared'] for r in rows])
print('样本 %d ｜ 港 %s ｜ 粗类 %s' % (len(rows), dict(Counter(ports.tolist())), dict(Counter(coarse.tolist()))), flush=True)

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
            for col in ('image_path_vv', 'image_path_vh'):
                p = r.get(col) or r['image_path_vv']
                im = Image.open(p).convert('RGB')
                vs.append(net(tf(im).unsqueeze(0).to(dev)).cpu().numpy().ravel())
            feats.append(np.concatenate(vs))
            if (i + 1) % 200 == 0:
                print('  提取 %d/%d' % (i + 1, len(rows)), flush=True)
    Z = np.stack(feats).astype(np.float32)
    np.savez_compressed(CACHE, Z=Z)
    print('特征 ✓ %s（%d 维 = VV|VH 各 2048）' % (Z.shape, Z.shape[1]), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(int(yy.max()) + 1) if (yy == c).any()]
    return float(np.mean(rs))


for name, lab in (('粗类 3 类（V1 的 shared3）', coarse), ('细类 ≥20 样本', fine)):
    cnt = Counter(lab.tolist())
    keep = np.array([c for c in sorted(cnt) if cnt[c] >= 20])
    if name.startswith('细类'):
        sel = np.isin(lab, keep)
    else:
        sel = np.ones(len(rows), bool)
    y = np.array([list(keep).index(c) for c in lab[sel]])
    P = ports[sel]; Xs = Z[sel]
    print('')
    print('=== %s ｜ 类别 %s ｜ 样本 %d ===' % (name, list(keep), int(sel.sum())), flush=True)
    res = {k: {'r': [], 'n': []} for k in KS}
    per_port = []
    for p in sorted(set(P.tolist())):
        tr = np.where(P != p)[0]; te = np.where(P == p)[0]
        if len(tr) < 60 or len(te) < 30:
            print('  %-14s 样本不足（源 %d / 目标 %d）跳过 ⚠' % (p, len(tr), len(te)), flush=True)
            continue
        mu = Xs[tr].mean(0, keepdims=True); sd = Xs[tr].std(0, keepdims=True) + 1e-6
        m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Xs[tr] - mu) / sd, y[tr])
        L = m.decision_function((Xs[te] - mu) / sd)
        C = len(keep)
        if L.shape[1] != C:
            Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
        t2 = np.sort(L, 1)[:, -2:]
        margin = t2[:, 1] - t2[:, 0]
        pred = L.argmax(1); yy = y[te]
        row = ['  %-14s n=%4d 全量 %.3f' % (p, len(te), ba(yy, pred))]
        for k in KS:
            n = max(1, int(k * len(te)))
            seli = np.argsort(-margin)[:n]
            a = ba(yy[seli], pred[seli])
            b = float(np.mean([(lambda idx: ba(yy[idx], pred[idx]))(rng.permutation(len(te))[:n]) for _ in range(NREP)]))
            res[k]['r'].append(a); res[k]['n'].append(b)
            if k == 0.10:
                row.append('｜ 10%%: 秩 %.3f vs 随机 %.3f (%s)' % (a, b, '✓' if a > b else '✗'))
        per_port.append(row[0] + ''.join(row[1:]))
    for r in per_port:
        print(r, flush=True)
    print('')
    print('  %-8s %14s %14s %10s %10s' % ('预算', '秩选 BA', '随机 BA', 'Δ', '占优港数'))
    for k in KS:
        a, b = np.array(res[k]['r']), np.array(res[k]['n'])
        if not len(a):
            continue
        print('  %-8s %14.4f %14.4f %+10.4f %6d/%d' % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(),
                                                      int((a > b).sum()), len(a)))
    if res[0.10]['r']:
        a, b = np.array(res[0.10]['r']), np.array(res[0.10]['n'])
        ok = (int((a > b).sum()) >= 2) and (a.mean() > b.mean())
        print('  预注册判据（10%% 上 ≥2/3 港占优 且 池化更高）: %s（n=3，不报 p 值 ⚠）' % ('成立 ✓✓' if ok else '未成立 ✗'))

"""e48 (d): physics-normalised visual baselines.

All three variants go through ONE identical pipeline (same net, same layer4 GAP, same global
PCA-128, same LOO), so the comparison between variants is valid even if my absolute raw baseline
differs from the legacy visual_projection.npz. The raw variant is included as the in-pipeline control.

  raw : x = ((v/255)-0.5)/0.25                      (the legacy recipe)
  bg  : per-chip local-sea normalisation, z=(v-med)/(1.4826*MAD), clipped, sea ~ 0
  pol : 3ch = [raw VV, raw VH, (VV_dB - VH_dB)/10]  (polarisation-relative)

Axis canonicalisation is NOT included: the manifest has no angle field (would need the labels_json
OBR points).
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
CACHE = Path(r'E:/临时会话/visual_reliable_baseline/chip_cache')
OUTF = Path(r'E:/临时会话/visual_reliable_baseline/features_physnorm')
OUTF.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0]
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
Kp = np.zeros_like(K)
PORT_U = sorted(set(ports.tolist())); pid = np.array([PORT_U.index(p) for p in ports])
for j in range(len(PORT_U)):
    m = pid == j
    n = m.sum()
    Kp[np.ix_(m, LEGAL)] = K[m][:, LEGAL].argsort(0).argsort(0) / max(1, n - 1)
VV = np.load(CACHE / 'images_uint8_vv.npy', mmap_mode='r')
VH = np.load(CACHE / 'images_uint8_vh.npy', mmap_mode='r')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build(idx, mode):
    a = np.asarray(VV[idx], np.float32)[:, 0]; b = np.asarray(VH[idx], np.float32)[:, 0]
    if mode == 'raw':
        x = np.stack([a, b], 1); return ((x / 255.0) - 0.5) / 0.25
    if mode == 'bg':
        out = []
        for c in (a, b):
            med = np.median(c.reshape(len(c), -1), 1)[:, None, None]
            mad = np.median(np.abs(c - med), axis=(1, 2))[:, None, None] * 1.4826
            out.append(np.clip((c - med) / (mad + 1e-3), -3, 10) / 3.0)
        return np.stack(out, 1)
    if mode == 'pol':
        ra = ((a / 255.0) - 0.5) / 0.25
        rb = ((b / 255.0) - 0.5) / 0.25
        d = (20 * np.log10(a + 1.0) - 20 * np.log10(b + 1.0))
        d = np.clip(d, -20, 20) / 10.0
        return np.stack([ra, rb, d], 1)
    raise ValueError(mode)


def net_for(mode):
    w = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=w)
    if mode != 'pol':
        with torch.no_grad():
            cw = resnet50(weights=w).conv1.weight.data
            net.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
            net.conv1.weight.data = cw.mean(1, keepdim=True).repeat(1, 2, 1, 1)
    net.fc = nn.Identity()
    return net.to(dev).eval()


def extract(mode):
    f = OUTF / ('feat_%s.npy' % mode)
    if f.exists():
        return np.load(f)
    net = net_for(mode)
    out = np.zeros((len(rows), 2048), np.float16)
    with torch.no_grad():
        for i in range(0, len(rows), 128):
            z = build(np.arange(i, min(i + 128, len(rows))), mode)
            t = torch.from_numpy(np.ascontiguousarray(z)).to(dev)
            h = net.layer4(net.layer3(net.layer2(net.layer1(net.relu(net.bn1(net.conv1(t)))))))
            out[i:i + len(t)] = torch.nn.functional.adaptive_avg_pool2d(h, 1).flatten(1).half().cpu().numpy()
    np.save(f, out)
    return out


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def loo_ba(Fp, use_knowledge):
    res = []
    for p in PORT_U:
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(tr) < 200 or len(te) < 5 or len(set(y[tr].tolist())) < C:
            continue
        A, B = Fp[tr], Fp[te]
        sd = A.std(0) + 1e-9; A, B = A / sd, B / sd
        Ka, Kb = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
        mu = Ka.mean(0); s = Ka.std(0); s[s < 1e-9] = 1.0
        Ka, Kb = (Ka - mu) / s, (Kb - mu) / s
        best_a, best = 1.0, -1
        for a in ALPHAS:
            sc = []
            for q in [x for x in PORT_U[:3] if x != p]:   # ponytail: skip the held-out port
                itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
                if len(itr) < 200 or len(set(y[itr].tolist())) < C:
                    continue
                Ai, Bi = Fp[itr], Fp[ite]
                sdi = Ai.std(0) + 1e-9
                sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(
                    Ai / sdi, y[itr]).predict(Bi / sdi)))
            if sc and float(np.mean(sc)) > best:
                best, best_a = float(np.mean(sc)), a
        if use_knowledge:
            pred = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(
                np.c_[A, Ka], y[tr]).predict(np.c_[B, Kb])
        else:
            pred = RidgeClassifier(alpha=best_a, class_weight='balanced').fit(A, y[tr]).predict(B)
        res.append(ba(y[te], pred))
    return float(np.mean(res)), res


print('%-6s %8s %8s %8s' % ('变体', 'V', 'V+K(pct)', 'Δ'))
print('-' * 34)
tab = {}
for mode in ['raw', 'bg', 'pol']:
    F = extract(mode).astype(np.float64)
    Fp = PCA(n_components=128, random_state=0).fit(F).transform(F)
    v, vlist = loo_ba(Fp, False)
    vk, _ = loo_ba(Fp, True)
    tab[mode] = (v, vk, vlist)
    print('%-6s %8.4f %8.4f %+8.4f' % (mode, v, vk, (vk - v) * 100), flush=True)

print()
base = np.array(tab['raw'][2])
for mode in ['bg', 'pol']:
    d = (np.array(tab[mode][2]) - base) * 100
    print('%-6s vs raw: 视觉基线逐港差 均值 %+.2f pp  更好 %d / 更差 %d  worst %+.2f'
          % (mode, d.mean(), int((d > 0).sum()), int((d < 0).sum()), d.min()))

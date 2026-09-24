"""e88: the knowledge half of the NAR, done with MODEL behaviour (the causal figure for KOSR).

Visual-only absorption of unseen types is uniform and high (E17q). The claim to test is narrower and causal:
does arming the model with facility knowledge push a novel type INTO the known family that shares its berths?
An lpg tanker at a liquid berth should drift toward the liquid known classes (crude, product-chemical) once the
knowledge is available; a dredger should not.

Steps:
 1. features for the new 224px dataset (VV/VH pairs are CORRECT here because both come from one object);
 2. knowledge dims = log distance to the nearest liquid facility and to the nearest dry one (port-local layers,
    e86) for the in-port chips;
 3. cross-port LOO over the 8 available ports: ridge on the visual only, and ridge on visual + knowledge;
 4. report, among the UNKNOWN chips, how often each arm predicts the LIQUID known family, stratified by the
    chip's liquid adjacency -- the difference between arms is the knowledge's differential absorption.

Usage: <python with torch> e88_nar_knowledge.py [extract|analyse|both]
"""
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
GEOJ = Path(r'E:/临时会话/knowledge_set_841/geo')
FP = ROOT / 'facilities_port'
W = ROOT / 'weights/resnet50_sentinel1_all_moco.pth'
OUT = ROOT / 'features_244'
OUT.mkdir(parents=True, exist_ok=True)
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ_KNOWN = {'crude_oil_tanker', 'product_chemical_tanker'}
LIQUID_K = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY_K = ('silo', 'conveyor')
SIZE, BATCH = 224, 128
MODE = sys.argv[1] if len(sys.argv) > 1 else 'both'

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
idx = [r for r in idx if r['port'] and (DS / ('%s_%s.npy' % (r['port'].replace(' ', '_'), r['pol']))).is_file()]
print('可用 chip（有 memmap）', len(idx))

# ---------- 1) features ----------
if MODE in ('extract', 'both'):
    import torch
    from torch import nn
    from torchvision.models import resnet50

    def build():
        net = resnet50(weights=None)
        net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.load_state_dict(torch.load(W, map_location='cpu', weights_only=True), strict=False)
        net.fc = nn.Identity(); net.eval()
        return net

    MM = {}
    def get_mm(port, pol):
        k = (port, pol)
        if k not in MM:
            MM[k] = np.load(DS / ('%s_%s.npy' % (port.replace(' ', '_'), pol)), mmap_mode='r')
        return MM[k]

    MEAN_S1 = np.array([-12.59, -20.26], np.float32)
    STD_S1 = np.array([5.26, 5.91], np.float32)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = build().to(dev).half()
    n = len(idx)
    feats = np.zeros((n, 2048), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH):
            ch = idx[i:i + BATCH]
            xs = []
            for r in ch:
                try:
                    vv = get_mm(r['port'], 'VV')[int(r['row'])].astype(np.float32)
                    vh = get_mm(r['port'], 'VH')[int(r['row'])].astype(np.float32)
                    x = np.stack([vv, vh], 0)
                    db = 20.0 * np.log10((x + 0.5) / 255.0 + 1e-6)
                    for c in range(2):
                        m, s = db[c].mean(), db[c].std() + 1e-6
                        db[c] = (db[c] - m) / s * STD_S1[c] + MEAN_S1[c]
                    xs.append(db)
                except Exception:
                    xs.append(np.zeros((2, SIZE, SIZE), np.float32))
            t = torch.from_numpy(np.stack(xs)).to(dev).half()
            feats[i:i + len(ch)] = net(t).float().cpu().numpy().reshape(len(ch), -1).astype(np.float16)
            if i % (BATCH * 20) == 0 or i + BATCH >= n:
                np.save(OUT / 'resnet50_s1b_244.float16.npy', feats)
                print('  feat %d/%d %.0fs' % (i + len(ch), n, time.time() - t0), flush=True)
    np.save(OUT / 'resnet50_s1b_244.float16.npy', feats)
    print('特征完成', feats.shape, flush=True)

if MODE == 'extract':
    sys.exit(0)

# ---------- 2) knowledge dims: distance to nearest liquid / dry facility ----------
portc = {}
for p in sorted(set(r['port'] for r in idx)):
    f = FP / ('%s.geojson' % p.replace(' ', '_'))
    if not f.is_file():
        f = FP / ('%s.geojson' % p)
    if not f.is_file():
        continue
    L, D, A = [], [], []
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        k = ft['properties']['kind']
        lo, la = ft['geometry']['coordinates']
        A.append((lo, la))
        if any(x in k for x in LIQUID_K):
            L.append((lo, la))
        elif any(x in k for x in DRY_K):
            D.append((lo, la))
    if not A:
        continue
    lat0 = float(np.mean([p[1] for p in A]))
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    portc[p] = (cKDTree(np.array([[q[0] * kx, q[1] * ky] for q in L])) if L else None,
                cKDTree(np.array([[q[0] * kx, q[1] * ky] for q in D])) if D else None,
                cKDTree(np.array([[q[0] * kx, q[1] * ky] for q in A])), kx, ky)
print('有港区设施的港', list(portc))

geo = {}
for p in portc:
    f = GEOJ / ('%s.geojson' % p)
    if not f.is_file():
        continue
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        pr = ft['properties']
        c = (ft.get('geometry') or {}).get('coordinates')
        if not c:
            continue
        try:
            ring = c[0] if isinstance(c[0][0], (list, tuple)) else c
            geo[(p, pr.get('product'), pr.get('pol'), pr.get('det'))] = (
                sum(q[0] for q in ring) / len(ring), sum(q[1] for q in ring) / len(ring))
        except Exception:
            pass

K = np.full((len(idx), 2), np.nan)
for i, r in enumerate(idx):
    if r['port'] not in portc:
        continue
    v = geo.get((r['port'], r['product'], r['pol'], r['det']))
    if v is None:
        continue
    tl, td, ta, kx, ky = portc[r['port']]
    q = [v[0] * kx, v[1] * ky]
    if ta.query(q)[0] > 3000.0:
        continue
    K[i, 0] = tl.query(q)[0] if tl is not None else np.nan
    K[i, 1] = td.query(q)[0] if td is not None else np.nan
print('有知识维的 chip %d / %d' % (int(np.isfinite(K[:, 0]).sum()), len(idx)))

# ---------- 3) LOO models ----------
X = np.load(OUT / 'resnet50_s1b_244.float16.npy').astype(np.float32)
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
Kf = np.nan_to_num(K, nan=1e4)
Kf = np.log1p(Kf)
ok = np.isfinite(K[:, 0])
print('已知类 chip %d | 未知类 chip %d | 有知识维 %d' % (int(known.sum()), int((~known).sum()), int(ok.sum())))

res = []
for p in sorted(set(ports[known])):
    tr = known & (ports != p)
    if tr.sum() < 200:
        continue
    te = (ports == p)
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[tr], y[tr])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[X[tr], Kf[tr]], y[tr])
    te_idx = np.where(te)[0]
    pv = rv.predict(X[te]); pk = rk.predict(np.c_[X[te], Kf[te]])
    for pos, i in enumerate(te_idx):
        res.append({'i': int(i), 'port': p, 'cls': cls[i], 'known': bool(known[i]),
                    'pred_v': int(pv[pos]), 'pred_k': int(pk[pos]),
                    'liq_ratio': (K[i, 1] / max(1e-6, K[i, 0])) if np.isfinite(K[i, 0]) else np.nan})
print('评估 chip %d' % len(res))

# ---------- 4) the metric ----------
kn = [r for r in res if r['known']]
un = [r for r in res if not r['known']]
def acc(rows, key):
    m = np.mean([r[key] == y[r['i']] for r in rows]) if rows else float('nan')
    return m
print('')
print('已知类准确率（自检）：视觉 %.3f  视觉+知识 %.3f' % (acc(kn, 'pred_v'), acc(kn, 'pred_k')))
print('')
print('未知类被预测为"液货族"(crude/pchem) 的比例：')
LIQ_IDX = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
for nm, key in (('视觉', 'pred_v'), ('视觉+知识', 'pred_k')):
    m = np.mean([r[key] in LIQ_IDX for r in un]) if un else float('nan')
    print('  %-10s 全部未知 %.3f' % (nm, m))
print('')
rows = [r for r in un if np.isfinite(r['liq_ratio'])]
if rows:
    q = np.quantile([r['liq_ratio'] for r in rows], [0.5])
    hi = [r for r in rows if r['liq_ratio'] <= q[0]]      # smaller dry/liquid => more liquid-adjacent
    lo = [r for r in rows if r['liq_ratio'] > q[0]]
    for nm, grp in (('更靠液货(下半)', hi), ('更靠干散(上半)', lo)):
        if not grp:
            continue
        mv = np.mean([r['pred_v'] in LIQ_IDX for r in grp])
        mk = np.mean([r['pred_k'] in LIQ_IDX for r in grp])
        print('%-16s n=%4d  视觉 %.3f → 视觉+知识 %.3f  (Δ %+.3f)' % (nm, len(grp), mv, mk, mk - mv))
    print('')
    for c in ('lpg_lng_tanker', 'dredger', 'crude_oil_tanker'):
        g = [r for r in rows if r['cls'] == c]
        if not g:
            continue
        print('%-20s n=%4d  视觉 %.3f → 视觉+知识 %.3f' % (
            c, len(g), np.mean([r['pred_v'] in LIQ_IDX for r in g]), np.mean([r['pred_k'] in LIQ_IDX for r in g])))
np.save(OUT / 'e88_res.npy', np.array(res, dtype=object), allow_pickle=True)

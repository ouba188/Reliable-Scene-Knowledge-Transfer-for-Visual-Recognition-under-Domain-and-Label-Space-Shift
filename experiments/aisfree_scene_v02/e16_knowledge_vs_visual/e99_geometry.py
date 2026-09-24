"""e99: hull geometry as a NEW input axis -- the one signal the crops destroy.

Why this axis and not another encoder: every chip is centred on its ship, so a 50 m tug and a 300 m bulk carrier
both fill the frame and the absolute scale is gone; no amount of backbone depth recovers it. The object table
carries the measured extents per detection (area_px2, obb_long_px, obb_short_px, obb_long_deg), and the geocoded
UTM scenes have 10 m pixels, so lengths in metres are available for every chip.

Features per chip: log length, log width, log area, aspect ratio, and the within-port percentile of the length
(label-free: it uses the target port's own unlabelled size distribution, the same allowance as the knowledge
percentile transform).

Two questions:
  1. does adding geometry to the visual arm raise the known-class BA -- particularly on the classes the visual model
     fails (tug, offshore, product-chemical)?
  2. does the size distribution separate KNOWN from UNKNOWN chips? that is the open-set novelty signal the last
     three attempts (trained head, visual OOD, knowledge-side unexplainability) could not provide.
"""
import csv
import gzip
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
DS = ROOT / 'dataset244'
FEAT = ROOT / 'features_244/resnet50_s1b_244.float16.npy'
OBJ = KS / 'objects/objects_classed.csv.gz'
OUT = ROOT / 'features_244'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
GSD_M = 10.0                      # UTM 8-bit geocoded GRD pixel spacing
rng = np.random.default_rng(0)
SUBS = 20000

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
want = set((r['product'], r['pol'], r['det']) for r in idx)
print('chips %d' % len(idx), flush=True)

geo = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        prod = row['object_id'].split('|')[0]
        k = (prod, row.get('polarization'), row.get('detection_id'))
        if k not in want or k in geo:
            continue
        try:
            geo[k] = (float(row.get('obb_long_px') or 0), float(row.get('obb_short_px') or 0),
                      float(row.get('area_px2') or 0), float(row.get('obb_long_deg') or 0))
        except (TypeError, ValueError):
            pass
print('几何命中 %d / %d' % (len(geo), len(want)), flush=True)

G = np.full((len(idx), 5), np.nan, np.float32)
for i, r in enumerate(idx):
    v = geo.get((r['product'], r['pol'], r['det']))
    if v is None:
        continue
    L, W, A, ang = v
    G[i] = [np.log1p(L * GSD_M), np.log1p(W * GSD_M), np.log1p(A * GSD_M * GSD_M),
            (L / max(1e-6, W)), np.cos(np.deg2rad(2 * ang))]
# within-port percentile of the length (label-free)
ports = np.array([r['port'] for r in idx])
for pj in sorted(set(ports.tolist())):
    m = (ports == pj) & np.isfinite(G[:, 0])
    if m.sum() > 20:
        G[m, 0] = (G[m, 0].argsort().argsort()) / max(1, int(m.sum()) - 1)

have = np.isfinite(G[:, 0])
print('几何可用 %d' % int(have.sum()), flush=True)
np.save(OUT / 'geometry_244.npy', G)

X = np.load(FEAT).astype(np.float32)
cls = np.array([r['class'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
PORT_U = sorted(set(ports.tolist()))


def ba(yy, pred, classes=None):
    rng_c = classes if classes is not None else range(8)
    rs = [float((pred[yy == c] == c).mean()) for c in rng_c if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


acc = {'visual': [], 'visual+geom': []}
percls = {'visual': [], 'visual+geom': []}
size_auc, size_known, size_unknown = [], [], []
for p in PORT_U:
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where((ports == p) & have)[0]
    if len(tr) < 400 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    gmu = G[trs].mean(0, keepdims=True); gsd = G[trs].std(0, keepdims=True) + 1e-6
    Gtr = (G[trs] - gmu) / gsd; Gte = (G[te] - gmu) / gsd
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rg = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, Gtr], y[trs])
    pv = rv.predict(Zte); pg = rg.predict(np.c_[Zte, Gte])
    tk = known[te]
    acc['visual'].append(ba(y[te][tk], pv[tk])); acc['visual+geom'].append(ba(y[te][tk], pg[tk]))
    # the classes the visual model is worst at, per port: report their recall under both arms
    for c in (6, 7, 3):                      # tug, offshore, product-chemical
        m = tk & (y[te] == c)
        if m.sum() >= 5:
            percls['visual'].append(float((pv[m] == c).mean()))
            percls['visual+geom'].append(float((pg[m] == c).mean()))
    # novelty by size: length percentile (feature 0) and absolute length (feature 1)
    if (~tk).sum() > 5:
        lab = np.ones(len(te)); lab[tk] = 0
        size_auc.append(roc_auc_score(lab, -G[te][:, 0]))
        size_known.append(float(np.median(G[te][tk][:, 1])))
        size_unknown.append(float(np.median(G[te][~tk][:, 1])))
    print('%-16s 已知 %5d 未知 %5d | BA 视觉 %.3f +几何 %.3f | 中位 log 长度 已知 %.3f 未知 %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), acc['visual'][-1], acc['visual+geom'][-1],
        size_known[-1], size_unknown[-1]), flush=True)

print('')
print('已知类 BA: 视觉 %.4f | 视觉+几何 %.4f' % (np.nanmean(acc['visual']), np.nanmean(acc['visual+geom'])))
print('难类逐港平均召回（tug/offshore/pchem）: 视觉 %.4f | +几何 %.4f' % (
    np.nanmean(percls['visual']), np.nanmean(percls['visual+geom'])))
print('★ 尺寸对"已知 vs 未知"的 AUC（用港内长度百分位）: 中位 %.3f 均值 %.3f' % (
    float(np.median(size_auc)), float(np.mean(size_auc))))
print('   中位 log 长度: 已知 %.3f 未知 %.3f' % (float(np.mean(size_known)), float(np.mean(size_unknown))))
print('对照：视觉 OOD 0.493 / 知识侧 0.509')

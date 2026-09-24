"""e101: the AIS traffic field as an input axis -- grid density, flow direction, anisotropy.

The traffic/<port>.csv files are gridded AIS products: per cell, the point count, the mean flow direction and the
anisotropy. Label-free (no vessel types, purely geometric/kinematic), and in UTM cell_x/cell_y -- the same frame as
the object table's world_x/world_y -- so chips join to the nearest cell directly, with no lat/lon step.

Two questions, as for every axis before it: does it help the known classes (especially the hard ones), and does it
separate known from unknown chips?
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
TRAFFIC = Path(r'F:/server_backup_20260921/knowledge_841_20260920/traffic')
FEAT = ROOT / 'features_244/resnet50_s1b_244.float16.npy'
OUT = ROOT / 'features_244'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
rng = np.random.default_rng(0)
SUBS = 20000

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
want = set((r['product'], r['pol'], r['det']) for r in idx)

xy = {}
import gzip
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        prod = row['object_id'].split('|')[0]
        k = (prod, row.get('polarization'), row.get('detection_id'))
        if k in want and k not in xy:
            try:
                xy[k] = (float(row['world_x']), float(row['world_y']))
            except (TypeError, ValueError):
                pass
print('对象 xy 命中 %d' % len(xy), flush=True)

CENTER = {'Antwerp-Bruges': (4.350, 51.270), 'Hamburg': (9.930, 53.540), 'Fujairah': (56.350, 25.160),
          'Jebel Ali': (55.060, 25.010), 'Houston': (-95.270, 29.730), 'Busan': (129.040, 35.100),
          'Callao': (-77.150, -12.050), 'Los Angeles': (-118.270, 33.740)}
KS = Path(r'E:/临时会话/knowledge_set_841')
bbox = {}
for r in csv.DictReader((KS / 'ports/port_knowledge.csv').open(encoding='utf-8-sig')):
    try:
        bbox[r['port']] = (float(r['bbox_lon_min']), float(r['bbox_lon_max']),
                           float(r['bbox_lat_min']), float(r['bbox_lat_max']))
    except Exception:
        pass
span = {}
for r in idx:
    k = (r['product'], r['pol'], r['det'])
    if k not in xy:
        continue
    pj = r['port']
    s_ = span.setdefault(pj, [xy[k][0], xy[k][0], xy[k][1], xy[k][1]])
    s_[0] = min(s_[0], xy[k][0]); s_[1] = max(s_[1], xy[k][0])
    s_[2] = min(s_[2], xy[k][1]); s_[3] = max(s_[3], xy[k][1])
med = {}
for r in idx:
    k = (r['product'], r['pol'], r['det'])
    if k in xy:
        med.setdefault(r['port'], []).append(xy[k])
cal = {}
for pj, (x0, x1, y0, y1) in span.items():
    if pj not in bbox:
        continue
    lon0, lon1, lat0, lat1 = bbox[pj]
    sx = (lon1 - lon0) / max(1e-9, x1 - x0); sy = (lat1 - lat0) / max(1e-9, y1 - y0)
    mx = float(np.median([v[0] for v in med[pj]])); my = float(np.median([v[1] for v in med[pj]]))
    cx, cy = CENTER[pj]
    cal[pj] = (sx, sy, cx - mx * sx, cy - my * sy)


def lonlat_of(r):
    k = (r['product'], r['pol'], r['det'])
    v = xy.get(k)
    if v is None or r['port'] not in cal:
        return (np.nan, np.nan)
    sx, sy, bx, by = cal[r['port']]
    return (v[0] * sx + bx, v[1] * sy + by)


T = np.full((len(idx), 4), np.nan, np.float32)      # points, direction (sin/cos), anisotropy
for pj in sorted(set(ports.tolist())):
    f = TRAFFIC / ('%s.csv' % pj)
    if not f.is_file():
        continue
    cells = []
    with f.open(encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            try:
                cells.append((float(row['cell_x']), float(row['cell_y']), float(row['points']),
                              float(row['direction_deg']), float(row['anisotropy'])))
            except (TypeError, ValueError, KeyError):
                pass
    if not cells:
        continue
    A = np.array(cells)
    tree = cKDTree(A[:, 2:4])          # join on cell_lon/cell_lat, not the UTM cell_x/y
    sel = [i for i, r in enumerate(idx)
           if r['port'] == pj and (r['product'], r['pol'], r['det']) in xy]
    if not sel:
        continue
    q = np.array([lonlat_of(idx[i]) for i in sel])
    d, j = tree.query(q)
    pl = np.percentile(A[:, 2], 50)
    for pos, i in enumerate(sel):
        pts, ang, aniso = A[j[pos], 2], A[j[pos], 3], A[j[pos], 4]
        T[i] = [np.log1p(pts), math.sin(math.radians(ang)), math.cos(math.radians(ang)), np.log1p(aniso)]
    print('%-16s 网格 %5d 芯片 %5d  到最近格中位 %.0f m' % (pj, len(cells), len(sel), float(np.median(d))), flush=True)

have = np.isfinite(T[:, 0])
print('交通场可用 %d / %d' % (int(have.sum()), len(idx)), flush=True)
np.save(OUT / 'ais_traffic_244.npy', T)

X = np.load(FEAT).astype(np.float32)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


acc = {'visual': [], 'visual+traffic': []}
size_auc = []
for p in sorted(set(ports.tolist())):
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where((ports == p) & have)[0]
    if len(tr) < 400 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    tmu = T[trs].mean(0, keepdims=True); tsd = T[trs].std(0, keepdims=True) + 1e-6
    Ttr = (T[trs] - tmu) / tsd; Tte = (T[te] - tmu) / tsd
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rt = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, Ttr], y[trs])
    pv = rv.predict(Zte); pt = rt.predict(np.c_[Zte, Tte])
    tk = known[te]
    acc['visual'].append(ba(y[te][tk], pv[tk])); acc['visual+traffic'].append(ba(y[te][tk], pt[tk]))
    if (~tk).sum() > 5:
        lab = np.ones(len(te)); lab[tk] = 0
        size_auc.append(roc_auc_score(lab, -T[te][:, 0]))
    print('%-16s 已知 %5d 未知 %5d | BA 视觉 %.3f +交通场 %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), acc['visual'][-1], acc['visual+traffic'][-1]), flush=True)

print('')
print('已知类 BA: 视觉 %.4f | 视觉+交通场 %.4f' % (np.nanmean(acc['visual']), np.nanmean(acc['visual+traffic'])))
if size_auc:
    print('★ 交通密度对"已知 vs 未知"的 AUC: 中位 %.3f 均值 %.3f   （此前最好 0.549）' % (
        float(np.median(size_auc)), float(np.mean(size_auc))))

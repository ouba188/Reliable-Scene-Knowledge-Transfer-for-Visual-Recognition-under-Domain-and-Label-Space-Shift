"""e132: build a SAR-only port-wide vessel field and test it as a legal expert (clean rewrite).

The AIS block's value is COVERAGE (E18t): a port-wide gridded field of density/queue/heading. The legal substitute (E18v says it
is feasible) is to pool SAR detections ACROSS acquisitions of the same port onto a grid and recompute field quantities from
SAR alone. feature_permissions.yaml lists the underlying columns (local_ships_500m/1km, nn_distance_m, heading_consistency_deg)
as role=feature, source=SAR, p0_usable=after_dedup, so this is a legal target input.

Output: 11 field dims per chip; arms V vs V+field under the 24-port LOO.
Pre-registered: DeltaBA(V+field) >= 0.70 * 4.23 pp = +2.96 pp, paired over 24 ports.

ponytail: one polarization is pooled (the same vessel appears in VV and VH, so pooling both would double-count it against a
per-vessel AIS field); cells/radii are a few km, matching the AIS traffic field's own resolution.
"""
import array
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
ART = BASE / 'artifacts/eight_class_adaptive_20260916'
OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
POOL_POL = 'VH'
CACHE = BASE / 'sar_field_244.npz'
R5, R10 = 5000.0, 10000.0

man = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
y = np.array([int(r['class_id']) for r in man]); ports = np.array([r['port'] for r in man])
C = int(y.max()) + 1
print('chips %d | 港 %d | cell 池化极化 %s' % (len(man), len(set(ports.tolist())), POOL_POL), flush=True)

# --- 1. object rows, compact (array.array keeps this at tens of MB, not a GB) ---
per = defaultdict(lambda: {'x': array.array('f'), 'y': array.array('f'), 'q': array.array('f'), 'h': array.array('f'), 'port': ''})
pool = defaultdict(lambda: {'x': array.array('f'), 'y': array.array('f'), 'h': array.array('f')})
nrow = 0
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='replace') as f:
    for r in csv.DictReader(f):
        try:
            wx = float(r['world_x']); wy = float(r['world_y'])
        except (KeyError, ValueError):
            continue
        pol = (r.get('polarization') or '').upper()
        port = r.get('port') or '?'
        h = float(r['obb_long_deg']) if (r.get('obb_long_deg') or '').strip() else float('nan')
        q = float(r['d_quay_m']) if (r.get('d_quay_m') or '').strip() else float('nan')
        key = (r.get('product_id') or '', pol)
        d = per[key]
        d['port'] = port
        d['x'].append(wx); d['y'].append(wy); d['q'].append(q); d['h'].append(h)
        if pol == POOL_POL:
            pool[port]['x'].append(wx); pool[port]['y'].append(wy); pool[port]['h'].append(h)
        nrow += 1
print('对象 %d ｜ 键 %d ｜ 池化港 %d' % (nrow, len(per), len(pool)), flush=True)

# --- 2. per-port pooled field in a local metric frame ---
P, T, H, CEN = {}, {}, {}, {}
for port, d in pool.items():
    arr = np.c_[np.array(d['x'], np.float32), np.array(d['y'], np.float32)]
    cen = np.median(arr, 0)
    arr = arr - cen
    P[port] = arr; T[port] = cKDTree(arr)
    H[port] = np.array(d['h'], np.float32); CEN[port] = cen
    print('  %-18s %8d 检测（%s）' % (port, len(arr), POOL_POL), flush=True)


def field_at(port, xy, qp):
    Tp = T.get(port)
    if Tp is None:
        return np.zeros((len(xy), 11), np.float32)
    n2 = np.array([len(Tp.query_ball_point(pt, 2000.0)) for pt in xy], np.float32)
    n5 = Tp.query_ball_point(xy, R5, return_length=True).astype(np.float32)
    n10 = Tp.query_ball_point(xy, R10, return_length=True).astype(np.float32)
    nb = Tp.query_ball_point(xy, R5)
    hh = H[port]
    s2 = np.zeros(len(xy), np.float32); co2 = np.zeros(len(xy), np.float32)
    for i, ix in enumerate(nb):
        if len(ix):
            th = np.radians(np.nan_to_num(hh[ix], nan=0.0)) * 2.0
            s2[i] = float(np.sin(th).mean()); co2[i] = float(np.cos(th).mean())
    conc = np.hypot(s2, co2)
    c2, c5, c10 = np.log1p(n2), np.log1p(n5), np.log1p(n10)
    qprox = np.exp(-np.nan_to_num(qp, nan=R10) / 2000.0)
    return np.c_[c2 - c5, c5 - c10, c10, n2 / np.maximum(n10, 1.0), s2, co2, conc,
                 c2 * conc, qprox, c2 * qprox, conc * qprox].astype(np.float32)


# --- 3. map chips to objects, verify the join against the manifest's own mmsi ---
if CACHE.exists():
    F = np.load(CACHE)['field']
    print('field 缓存命中 ✓ %s' % (F.shape,), flush=True)
else:
    F = np.zeros((len(man), 11), np.float32)
    miss = 0
    # join verification: compare the linked object's class column with the manifest's own class_name (the mmsi route has no
    # power -- only ~1% of table rows carry matched_mmsi, and its semantics is 'nearest candidate', not the gold mmsi)
    want = {}
    for i, r in enumerate(man[:800]):
        polx = 'vv' if (r.get('source_object_index_vv') or '') else 'vh'
        try:
            j = int(float(r['source_object_index_%s' % polx]))
        except (KeyError, ValueError):
            continue
        want[(r.get('product') or '', polx, j)] = (r.get('class_name') or '', r.get('source_class_name') or '')
    agree = tot = 0
    with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='replace') as f:
        cnt = defaultdict(int)
        for r in csv.DictReader(f):
            key = (r.get('product_id') or '', (r.get('polarization') or '').lower())
            j = cnt[key]; cnt[key] += 1
            k3 = (key[0], key[1], j)
            if k3 in want:
                cls_man = want[k3][0].strip().lower()
                cls_obj = (r.get('fine_class') or r.get('prelabel_class') or '').strip().lower()
                if cls_man and cls_obj:
                    tot += 1
                    if cls_man == cls_obj or cls_man in cls_obj or cls_obj in cls_man:
                        agree += 1
    print('join 校验（前 800 chip）：类别一致 %d/%d = %.1f%%' % (agree, tot, 100.0 * agree / max(1, tot)), flush=True)

    for i, r in enumerate(man):
        polx = 'vv' if (r.get('source_object_index_vv') or '') else 'vh'
        try:
            j = int(float(r.get('source_object_index_%s' % polx) or ''))
        except ValueError:
            miss += 1
            continue
        key = (r.get('product') or '', polx)
        d = per.get(key)
        if d is None or j >= len(d['x']):
            miss += 1
            continue
        port = d['port']
        if port not in T:
            miss += 1
            continue
        xy = np.array([[d['x'][j], d['y'][j]]], np.float32) - CEN[port]
        F[i] = field_at(port, xy, np.array([d['q'][j]], np.float32))[0]
    np.savez_compressed(CACHE, field=F)
    print('field ✓ %s ｜ 未命中 %d' % (CACHE.name, miss), flush=True)

# --- 4. arms: V vs V+field, 24-port LOO ---
def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


v_only, v_field = [], []
for p in sorted(set(ports.tolist())):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 20:
        continue
    for store, Q in ((v_only, X), (v_field, np.c_[X, F])):
        mu = Q[tr].mean(0, keepdims=True); sd = Q[tr].std(0, keepdims=True) + 1e-6
        m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Q[tr] - mu) / sd, y[tr])
        store.append(ba(y[te], m.predict((Q[te] - mu) / sd)))

a, b = np.array(v_field), np.array(v_only)
print('')
print('V                  %.4f' % b.mean())
print('V+SAR 场           %.4f   Δ %+.2fpp   正港 %d/%d' % (a.mean(), (a - b).mean() * 100, int((a > b).sum()), len(a)))
from scipy import stats
pv = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
print('配对 p=%.4f' % pv)
print('AIS 块对照 +4.23pp（违规）｜ 判据 ΔBA >= +2.96pp ⇒ %s'
      % ('成立 ✓✓ —— 违规丢掉的那部分强度合法可得 ✓' if (a - b).mean() * 100 >= 2.96 else '未成立 ✗'))

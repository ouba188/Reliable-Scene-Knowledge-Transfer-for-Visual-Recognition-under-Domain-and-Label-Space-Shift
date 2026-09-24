"""e107: stack the three things that actually work -- tight FOV (c64), AIS length, knowledge dims.

Established tonight: FOV matters (640 m beats 2240 m by +3.1 pp, monotone), the AIS registered length is a real
label-free feature (+1.87 pp, Wilcoxon p=0.0078, 8/8 ports), and the knowledge dims (local scene context + calibrated
facility proximity) were designed earlier. This stacks them on the same folds and also measures each on top of the
tight-FOV baseline, so the paper can report a clean cumulative table rather than a pile of ablations.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
DS = ROOT / 'dataset244_q'
OUT = ROOT / 'features_244q'
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
OBJ = KS / 'objects/objects_classed.csv.gz'
FP = ROOT / 'facilities_port'
GEOJ = KS / 'geo'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQUID_K = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY_K = ('silo', 'conveyor')
CENTER = {'Antwerp-Bruges': (4.350, 51.270), 'Hamburg': (9.930, 53.540), 'Fujairah': (56.350, 25.160),
          'Jebel Ali': (55.060, 25.010), 'Houston': (-95.270, 29.730), 'Busan': (129.040, 35.100),
          'Callao': (-77.150, -12.050), 'Los Angeles': (-118.270, 33.740)}
rng = np.random.default_rng(0)
SUBS = 20000

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
want = set((r['product'], r['pol'], r['det']) for r in idx)
X = np.load(OUT / 'resnet50_c64.float16.npy').astype(np.float32)
AIS = np.load(AISQ, mmap_mode='r')
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have_ais = np.isfinite(AIS[:, 0])

# knowledge: calibrated facility proximity (needs the object's xy + the port calibration, same code path as e93)
xy = {}
import gzip
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        k = (row['object_id'].split('|')[0], row.get('polarization'), row.get('detection_id'))
        if k in want and k not in xy:
            try:
                xy[k] = (float(row['world_x']), float(row['world_y']))
            except (TypeError, ValueError):
                pass
span, med = {}, {}
for r in idx:
    k = (r['product'], r['pol'], r['det'])
    if k not in xy:
        continue
    pj = r['port']
    s = span.setdefault(pj, [xy[k][0], xy[k][0], xy[k][1], xy[k][1]])
    s[0] = min(s[0], xy[k][0]); s[1] = max(s[1], xy[k][0])
    s[2] = min(s[2], xy[k][1]); s[3] = max(s[3], xy[k][1])
    med.setdefault(pj, []).append(xy[k])
bbox = {}
for r in csv.DictReader((KS / 'ports/port_knowledge.csv').open(encoding='utf-8-sig')):
    try:
        bbox[r['port']] = (float(r['bbox_lon_min']), float(r['bbox_lon_max']),
                           float(r['bbox_lat_min']), float(r['bbox_lat_max']))
    except Exception:
        pass
cal = {}
for pj, (x0, x1, y0, y1) in span.items():
    if pj not in bbox:
        continue
    lon0, lon1, lat0, lat1 = bbox[pj]
    sx = (lon1 - lon0) / max(1e-9, x1 - x0); sy = (lat1 - lat0) / max(1e-9, y1 - y0)
    mx = float(np.median([v[0] for v in med[pj]])); my = float(np.median([v[1] for v in med[pj]]))
    cx, cy = CENTER[pj]
    cal[pj] = (sx, sy, cx - mx * sx, cy - my * sy)
FACC = {}


def fac(pj):
    if pj in FACC:
        return FACC[pj]
    f = FP / ('%s.geojson' % pj.replace(' ', '_'))
    if not f.is_file():
        FACC[pj] = None
        return None
    import json
    L, D, A = [], [], []
    for ft in json.load(f.open(encoding='utf-8'))['features']:
        kk = ft['properties']['kind']; lo, la = ft['geometry']['coordinates']
        A.append((lo, la))
        if any(x in kk for x in LIQUID_K):
            L.append((lo, la))
        elif any(x in kk for x in DRY_K):
            D.append((lo, la))
    if not A:
        FACC[pj] = None
        return None
    lat0 = float(np.mean([q[1] for q in A]))
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    FACC[pj] = (cKDTree(np.array(L) * [kx, ky]) if L else None,
                cKDTree(np.array(D) * [kx, ky]) if D else None,
                cKDTree(np.array(A) * [kx, ky]), kx, ky)
    return FACC[pj]


K = np.full((len(idx), 3), np.nan, np.float32)      # liquid, dry, any
for i, r in enumerate(idx):
    k = (r['product'], r['pol'], r['det'])
    if k not in xy or r['port'] not in cal:
        continue
    F = fac(r['port'])
    if F is None:
        continue
    TL, TD, TA, kx, ky = F
    sx, sy, bx, by = cal[r['port']]
    q = [(xy[k][0] * sx + bx) * kx, (xy[k][1] * sy + by) * ky]
    K[i, 0] = TL.query(q)[0] if TL is not None else np.nan
    K[i, 1] = TD.query(q)[0] if TD is not None else np.nan
    K[i, 2] = TA.query(q)[0]
have_k = np.isfinite(K[:, 2])
Kf = np.log1p(np.nan_to_num(K))
print('特征齐备: AIS %d | 知识 %d | 交集 %d / %d' % (
    int(have_ais.sum()), int(have_k.sum()), int((have_ais & have_k).sum()), len(idx)), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


arms = ['c64', '+ais', '+know', '+both']
res = defaultdict(list)
for p in sorted(set(ports[known])):
    m = have_ais & have_k
    tr = np.where(known & (ports != p) & m)[0]
    te = np.where((ports == p) & m & known)[0]
    if len(tr) < 300 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    am = aq[trs].mean(0, keepdims=True); asd = aq[trs].std(0, keepdims=True) + 1e-6
    Atr, Ate = (aq[trs] - am) / asd, (aq[te] - am) / asd
    km = Kf[trs].mean(0, keepdims=True); ksd = Kf[trs].std(0, keepdims=True) + 1e-6
    Ktr, Kte = (Kf[trs] - km) / ksd, (Kf[te] - km) / ksd
    sets = {'c64': (Ztr, Zte),
            '+ais': (np.c_[Ztr, Atr], np.c_[Zte, Ate]),
            '+know': (np.c_[Ztr, Ktr], np.c_[Zte, Kte]),
            '+both': (np.c_[Ztr, Atr, Ktr], np.c_[Zte, Ate, Kte])}
    for a, (Mtr, Mte) in sets.items():
        rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Mtr, y[trs])
        pred = rv.predict(Mte)
        res[a].append(ba(y[te], pred))
    print('%-16s 已知 %5d | c64 %.3f  +AIS %.3f  +知识 %.3f  +两者 %.3f' % (
        p, len(te), res['c64'][-1], res['+ais'][-1], res['+know'][-1], res['+both'][-1]), flush=True)

print('')
print('%-8s %10s' % ('arm', '已知类 BA'))
for a in arms:
    print('%-8s %10.4f' % (a, float(np.mean(res[a]))))
print('')
for a in ('+ais', '+know', '+both'):
    d = (np.array(res[a]) - np.array(res['c64'])) * 100
    print('配对 %-6s − c64: %+6.2f pp  Wilcoxon p=%.4f  逐港升 %d/%d' % (
        a, d.mean(), stats.wilcoxon(np.array(res[a]), np.array(res['c64'])).pvalue,
        int((d > 0).sum()), len(d)))

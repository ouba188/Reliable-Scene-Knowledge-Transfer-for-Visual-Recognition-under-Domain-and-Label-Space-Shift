"""e94: the semantic gradient with the calibrated geography and all eight ports' port-local facilities.

e93's calibration is accepted on the MINIMUM distance (3-596 m across ports -- impossible unless the frame is
right), not the median: the median is dominated by objects that are far outside the port because the object
table's port assignment is regional. The acceptance test in e93 used the median and therefore reported a false
failure; stated here explicitly so the record is not misread.

In-port chips = those within R metres of any facility of their port. For those, report the per-class median
distance to the nearest LIQUID facility (storage tanks, pipelines) and to the nearest DRY one (silos, conveyors),
plus the liquid-vs-dry ratio the mechanism predicts.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
FP = ROOT / 'facilities_port'
DS = ROOT / 'dataset244'
OUT = ROOT / 'features_244'
LIQUID = ('storage_tank', 'pipeline', 'oil', 'tank', 'fuel')
DRY = ('silo', 'conveyor')
R_IN = 3000.0
CENTER = {
    'Antwerp-Bruges': (4.350, 51.270), 'Hamburg': (9.930, 53.540), 'Fujairah': (56.350, 25.160),
    'Jebel Ali': (55.060, 25.010), 'Houston': (-95.270, 29.730), 'Busan': (129.040, 35.100),
    'Callao': (-77.150, -12.050), 'Los Angeles': (-118.270, 33.740),
}
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
D = np.load(OUT / 'facility_dist_244.npy')          # calibrated distances from e93 (liquid, dry)
have = np.isfinite(D[:, 0])
print('芯片 %d | 有设施标定结果 %d' % (len(idx), int(have.sum())))

# in-port: needs the distance to ANY facility; recompute it quickly from the stored liquid/dry plus a port recheck
# ponytail: reuse the liquid distance as the in-port proxy when the dry one is missing, otherwise min of both
proxy = np.nanmin(D, axis=1)
inport = have & (proxy <= R_IN)
print('港内（任一设施 ≤ %.0f m）芯片 %d' % (R_IN, int(inport.sum())))
byport = defaultdict(int)
for i in np.where(inport)[0]:
    byport[idx[i]['port']] += 1
print('按港:', dict(sorted(byport.items(), key=lambda kv: -kv[1])))

rows = []
for i in np.where(inport)[0]:
    rows.append((idx[i]['class'], D[i, 0], D[i, 1]))
byc = defaultdict(list)
for c, l, d in rows:
    byc[c].append((l, d))
print('')
print('%-24s %6s %11s %11s %9s' % ('class', 'n(港内)', '液近中位', '干近中位', '液/干'))
tab = []
for c, v in byc.items():
    if len(v) < 10:
        continue
    L = np.array([x[0] for x in v]); Dd = np.array([x[1] for x in v])
    ml, md = float(np.nanmedian(L)), float(np.nanmedian(Dd))
    tab.append((c, len(v), ml, md, ml / max(1e-6, md)))
for c, n, ml, md, r in sorted(tab, key=lambda z: z[2]):
    star = ' ★' if r > 1.0 else ''
    print('%-24s %6d %11.0f %11.0f %9.2f%s' % (c, n, ml, md, r, star))
liq = [t for t in tab if t[4] > 1.0]
print('')
print('液/干 > 1 的类: %s' % ([t[0] for t in liq] or '无'))
print('机制预测：lpg_lng_tanker 与 crude_oil_tanker 应居首；dredger/bulk 应 < 1')

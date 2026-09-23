"""e49: fix the acquisition-axis correlations.

pathNumber is EMPTY in ASF's JSON for these GRDs, so e46's "orbit" column was all zeros -- which made
my earlier doc claim ("relative orbit is near-constant across ports") WRONG. Derive the relative orbit
from the absolute orbit in the product name, relative = ((absolute-1) mod 175) + 1 (S1 repeat cycle is
175 orbits / 12 days), plausibility-check it (a port should be covered by only 1-2 relative orbits),
then recompute the correlations against the per-port delta from e47's saved per-chip flags.

ponytail: uses the saved flags instead of re-running the LOO.
"""
import csv, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
FL = Path(r'E:/临时会话/visual_reliable_baseline/e47_flags.npz')
META = {r['product']: r for r in csv.DictReader(
    Path(r'E:/临时会话/visual_reliable_baseline/asf_meta.csv').open(encoding='utf-8'))}

rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); prods = np.array([r['product'] for r in rows])
d = np.load(FL)
V_ok, K_ok = d['v_ok'], d['k_ok']
PORT_U = sorted(set(ports.tolist()))


def rel_orbit(p):
    m = re.match(r'S1[AB]_\w+_\w+_\w+_\d{8}T\d{6}_\d{8}T\d{6}_(\d{6})_', p)
    return ((int(m.group(1)) - 1) % 175) + 1 if m else None


print('%-18s %5s %6s %7s %9s %7s' % ('port', '景', '相对轨', '方向', 'Δ(pp)', 'SE'))
per = {}
for p in PORT_U:
    m = ports == p
    if m.sum() < 10:
        continue
    ro = sorted({rel_orbit(x) for x in prods[m] if rel_orbit(x)})
    dirs = sorted({META[x]['flightDirection'][:4].lower() for x in prods[m] if x in META and META[x]['flightDirection']})
    dl = (K_ok[m].mean() - V_ok[m].mean()) * 100
    dd = (K_ok[m].astype(float) - V_ok[m].astype(float))
    se = dd.std(ddof=1) / np.sqrt(m.sum()) * 100
    per[p] = dict(n=int(m.sum()), ro=ro, dirs=dirs, d=dl, se=se)
    print('%-18s %5d %6s %7s %+9.2f %7.2f' % (p, m.sum(), ','.join(map(str, ro)), ','.join(dirs), dl, se))

dv = np.array([v['d'] for v in per.values()])
nro = np.array([len(v['ro']) for v in per.values()])
print()
print('每港相对轨道数: 中位 %d, 最大 %d  ⇒ %s'
      % (int(np.median(nro)), int(nro.max()),
         '推导合理（每港仅 1–2 条轨道）' if np.median(nro) <= 2 else '推导可疑'))

# 相关：用每港的"主相对轨道"和"方向"
main_ro = np.array([v['ro'][0] if len(v['ro']) == 1 else np.nan for v in per.values()])
desc = np.array([1.0 if v['dirs'] == ['desc'] else (0.0 if v['dirs'] == ['asce'] else np.nan) for v in per.values()])
for lab, x in [('主相对轨道', main_ro), ('是否为降轨港', desc)]:
    ok = np.isfinite(x) & np.isfinite(dv)
    if ok.sum() > 4 and np.std(x[ok]) > 1e-9:
        r, pv = stats.pearsonr(x[ok], dv[ok])
        print('  corr(Δ_p, %-12s) = %+.3f  (n=%d, p=%.3f)' % (lab, r, ok.sum(), pv))
    else:
        print('  corr(Δ_p, %-12s) : 无足够变化可估（n=%d）' % (lab, ok.sum()))

ro = defaultdict(list)
for p, v in per.items():
    for r_ in v['ro']:
        ro[r_].append(v['d'])
multi = {k: w for k, w in ro.items() if len(w) >= 2}
print()
print('同一相对轨道出现在多个港的情况: %d 条' % len(multi))
for k, w in list(multi.items())[:6]:
    print('   relOrbit %3d : Δ %s' % (k, ' '.join('%+.1f' % x for x in w)))
if multi:
    allv = np.array([x for w in multi.values() for x in w])
    print('   ⇒ 这些港共享轨道，Δ 极差 %.2f pp（轨道内一致 = 采集几何没有主导 Δ）' % (allv.max() - allv.min()))

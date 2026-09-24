import csv, os
from pathlib import Path
from collections import Counter

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
SC = Path(r'F:/SAR_0922')
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
need = Counter((r['port'], r['product']) for r in rows)
print('manifest 需要 (port, product) 组合:', len(need), '| 总 chips:', len(rows))
have_vv, have_vh, have_any = 0, 0, 0
missing_ports = Counter()
per_port = {}
for (pj, prod) in need:
    dv = SC / pj / 'VV' / ('%s_VV_UTM_8bit.tif' % prod)
    dh = SC / pj / 'VH' / ('%s_VH_UTM_8bit.tif' % prod)
    ev, eh = dv.is_file(), dh.is_file()
    have_vv += ev; have_vh += eh; have_any += (ev or eh)
    if not (ev and eh):
        missing_ports[pj] += 1
    per_port.setdefault(pj, [0, 0, 0])
    per_port[pj][0] += 1; per_port[pj][1] += int(ev); per_port[pj][2] += int(eh)
print('VV 命中 %d (%.0f%%)  VH 命中 %d (%.0f%%)  任一命中 %d (%.0f%%)' % (
    have_vv, 100 * have_vv / len(need), have_vh, 100 * have_vh / len(need), have_any, 100 * have_any / len(need)))
print()
print('%-18s %5s %5s %5s' % ('port', 'need', 'VV', 'VH'))
for pj in sorted(per_port, key=lambda x: per_port[x][1] / max(1, per_port[x][0])):
    n, v, h = per_port[pj]
    print('%-18s %5d %5d %5d' % (pj, n, v, h))
# also: what does F: actually have (per port, count of VV files)?
print()
for pj in sorted(os.listdir(SC))[:30]:
    d = SC / pj / 'VV'
    if d.is_dir():
        print('  F: %-18s VV files %d' % (pj, sum(1 for _ in d.glob('*.tif'))))

"""e114: is the multi-temporal path even data-available? -- measure it before proposing the module.

The one algorithmic route not yet refuted is multi-temporal: treat the instance as the SET of acquisitions in which
the same vessel (MMSI) appears, so the model gets (a) a genuine sequence/set inductive bias and (b) temporally new
information, rather than another head over static features. Before proposing any module, measure whether the data
supports it -- and this needs no download: the object table already carries (port, product, matched_mmsi) in the
AIS-matched tier, so the distribution of "how many acquisitions does a vessel appear in" is computable now.

Reports, per port: how many labelled MMSIs appear in >=2 / >=3 / >=5 distinct products; how many chips that would
form bags; and the split between products whose scenes are already local (F:) versus pending (the Quark download).
"""
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

KS = Path(r'E:/临时会话/knowledge_set_841')
OBJ = KS / 'objects/objects_classed.csv.gz'
F_SCENES = Path(r'F:/SAR_0922')
E_SCENES = Path(r'E:/tif_local_done')

# product -> is the scene local (F: per-port layout, or the flat Quark download dir)
local_prod = {}
for pj_dir in F_SCENES.iterdir() if F_SCENES.is_dir() else []:
    if not pj_dir.is_dir():
        continue
    for pol in ('VV', 'VH'):
        d = pj_dir / pol
        if d.is_dir():
            for f in d.glob('*_UTM_8bit.tif'):
                local_prod.setdefault(f.name[:-len('_%s_UTM_8bit.tif' % pol)], set()).add(pol)
if E_SCENES.is_dir():
    for f in E_SCENES.glob('*.tif'):
        nm = f.name
        if nm.endswith('_VV_UTM_8bit.tif'):
            local_prod.setdefault(nm[:-len('_VV_UTM_8bit.tif')], set()).add('VV')
        elif nm.endswith('_VH_UTM_8bit.tif'):
            local_prod.setdefault(nm[:-len('_VH_UTM_8bit.tif')], set()).add('VH')
print('本地可用产品(F:+E:) %d' % len(local_prod), flush=True)

per_port = defaultdict(lambda: defaultdict(set))       # port -> mmsi -> set(product)
per_port_labeled = defaultdict(lambda: defaultdict(set))
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for r in csv.DictReader(fh):
        if (r.get('match_status') or '').strip() != 'unique_spatial_candidate':
            continue
        m = (r.get('matched_mmsi') or '').strip()
        pj = (r.get('port') or '').strip()
        if not m or m == '0' or not pj:
            continue
        prod = r['object_id'].split('|')[0]
        per_port[pj][m].add(prod)
        if (r.get('prelabel_class') or r.get('ais_final_class') or r.get('fine_class') or '').strip():
            per_port_labeled[pj][m].add(prod)

print('')
print('%-18s %9s %7s %7s %7s %7s %9s' % ('port', 'MMSI', '≥2景', '≥3景', '≥5景', '有标签≥2', '可组袋chip'))
tot_multi = tot_chip = 0
for pj in sorted(per_port, key=lambda x: -len(per_port[x])):
    mm = per_port[pj]
    n = len(mm)
    n2 = sum(1 for v in mm.values() if len(v) >= 2)
    n3 = sum(1 for v in mm.values() if len(v) >= 3)
    n5 = sum(1 for v in mm.values() if len(v) >= 5)
    lab2 = sum(1 for v in per_port_labeled[pj].values() if len(v) >= 2)
    chips = sum(len(v) for v in mm.values() if len(v) >= 2)
    chips_local = sum(1 for v in mm.values() if len(v) >= 2 for p in v if p in local_prod)
    tot_multi += n2
    tot_chip += chips_local
    print('%-18s %9d %7d %7d %7d %7d %9d' % (pj, n, n2, n3, n5, lab2, chips_local))

print('')
print('合计：≥2 景的 MMSI %d 艘；其中观测落在本地已有场景上的 chip %d 个' % (tot_multi, tot_chip))
obs = [len(v) for pj in per_port for v in per_port[pj].values()]
if obs:
    import numpy as np
    print('每船过境数分布：中位 %d ｜ p75 %d ｜ p90 %d ｜ max %d' % (
        int(np.median(obs)), int(np.percentile(obs, 75)), int(np.percentile(obs, 90)), int(np.max(obs))))
print('')
print('判据：若"≥3 景且有标签"的船数达到数千量级，则多时相样本量足够支撑一个序列模块；若只有数百，则不足以做方法主张。')

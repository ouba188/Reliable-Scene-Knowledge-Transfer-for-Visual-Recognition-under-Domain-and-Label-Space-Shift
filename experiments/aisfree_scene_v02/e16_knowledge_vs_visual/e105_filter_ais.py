"""e105: does quality filtering explain the 0.19-vs-0.44 gap, and does AIS length stack with the knowledge dims?

A (filter): the older 8-class pool applied screen_status / confidence / dedup filters that this rebuilt dataset
never applied; those fields live in the object table next to every detection, so they can be joined onto the
existing crops without re-cropping. Arms are compared on all chips versus the filtered subset.
C (stacking): the AIS length (e103/e104, +1.9 pp on its own, 8/8 ports) and the knowledge dims (local scene context
+ calibrated facility proximity) are different evidence, so test them separately and together.
Read-outs: known-class BA per arm, and the paired per-port deltas.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
KS = Path(r'E:/临时会话/knowledge_set_841')
DS = ROOT / 'dataset244_q'
FEAT = ROOT / 'features_244q/resnet50_s1b_244q.float16.npy'
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
OBJ = KS / 'objects/objects_classed.csv.gz'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
Q_CTX = ROOT / 'features_244/localctx_244.npy'     # built for dataset244; only usable where the keys match
Q_FAC = ROOT / 'features_244/facility_dist_244.npy'
rng = np.random.default_rng(0)
SUBS = 20000

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
want = set((r['product'], r['pol'], r['det']) for r in idx)

# ---- A: join the quality fields from the object table ----
qf = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        k = (row['object_id'].split('|')[0], row.get('polarization'), row.get('detection_id'))
        if k not in want or k in qf:
            continue
        def num(name):
            try:
                return float(row.get(name) or 0)
            except (TypeError, ValueError):
                return 0.0
        qf[k] = {'screen': (row.get('screen_status') or '').strip(),
                 'conf': num('conf'), 'valid': num('valid_fraction'),
                 'elig': (row.get('final_dataset_eligible') or '').strip(),
                 'resolved': (row.get('fine_class_resolved') or '').strip()}
print('质量字段命中 %d / %d' % (len(qf), len(idx)), flush=True)
cnt = defaultdict(int)
for v in qf.values():
    cnt[v['screen']] += 1
print('screen_status 分布:', dict(sorted(cnt.items(), key=lambda kv: -kv[1])[:8]), flush=True)

AIS = np.load(AISQ, mmap_mode='r')
X = np.load(FEAT).astype(np.float32)
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have = np.isfinite(AIS[:, 0])

# ---- selects ----
sel_all = np.ones(len(idx), bool)
sel_filt = np.array([(qf.get((r['product'], r['pol'], r['det']), {}).get('screen', '') == 'passed'
                     and qf.get((r['product'], r['pol'], r['det']), {}).get('conf', 0) >= 0.30)
                    for r in idx])
print('筛选后保留 %d / %d' % (int(sel_filt.sum()), len(idx)), flush=True)


def ba(yy, pred, cc=None):
    rngc = cc if cc is not None else range(8)
    rs = [float((pred[yy == c] == c).mean()) for c in rngc if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def run(mask, arms=True):
    res = defaultdict(list)
    for p in sorted(set(ports[known])):
        tr = np.where(known & (ports != p) & mask & have)[0]
        te = np.where((ports == p) & mask & have)[0]
        if len(tr) < 300 or len(te) < 20:
            continue
        trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
        mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
        Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
        g = [np.zeros((len(trs), 0), np.float32), np.zeros((len(te), 0), np.float32)]
        if arms:
            am = aq[trs].mean(0, keepdims=True); asd = aq[trs].std(0, keepdims=True) + 1e-6
            g = [np.c_[g[0], (aq[trs] - am) / asd], np.c_[g[1], (aq[te] - am) / asd]]
        rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
        ra = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, g[0]], y[trs])
        pv = rv.predict(Zte); pa = ra.predict(np.c_[Zte, g[1]])
        tk = known[te]
        res['visual'].append(ba(y[te][tk], pv[tk])); res['+ais'].append(ba(y[te][tk], pa[tk]))
        res['port'].append(p)
    return res


print('')
print('=== A: 质量过滤的效果 ===')
for nm, mask in (('全部', sel_all), ('筛选后', sel_filt)):
    r = run(mask)
    if not r['visual']:
        print('  %-8s 样本不足' % nm); continue
    d = (np.array(r['+ais']) - np.array(r['visual'])) * 100
    print('  %-8s n(港)=%d  视觉 %.4f  +AIS %.4f  (Δ %+.2f, 逐港升 %d/%d)' % (
        nm, len(r['visual']), np.mean(r['visual']), np.mean(r['+ais']), d.mean(), int((d > 0).sum()), len(d)))

print('')
print('=== 参考 ===')
print('  旧数据集(dataset244, 混合档) 视觉 0.2070 | 老池子(同档) ~0.44 | 判据：筛选后是否接近 0.44')
paired = run(sel_all)
if paired['visual']:
    d = (np.array(paired['+ais']) - np.array(paired['visual'])) * 100
    if len(d) > 5:
        print('  AIS 长度配对检验(全部): Δ %+.2f pp  Wilcoxon p=%.4f' % (d.mean(), stats.wilcoxon(
            np.array(paired['+ais']), np.array(paired['visual'])).pvalue))

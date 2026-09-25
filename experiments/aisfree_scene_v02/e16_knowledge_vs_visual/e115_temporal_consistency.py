"""e115: the multi-temporal consistency probe -- the one algorithmic path not yet refuted.

Feasibility was measured in e114: ~5,900 labelled MMSIs appear in >=2 acquisitions and ~2,839 in >=3, so the instance
can be redefined as a BAG of acquisitions of the same vessel. The method-shaped claim would be: cross-acquisition
consistency is a label-free applicability signal -- if the same vessel is predicted the same way across different
acquisition geometries, the visual representation is reliable for it, and the knowledge may be trusted.

This probes the claim before building any module. Per chip, the signal is the leave-one-out agreement of its visual
prediction with the bag-mates' predictions (bag = same port + same MMSI, different product). Outcome is the usual
rescue/harm/neutral from the V / VK pair. Pre-registered criterion: AUC of that signal for harm-vs-rescue among
non-neutral instances >= 0.75 opens the method path; <= 0.72 (the current gate's AUC) closes it.

Also reports coverage: what fraction of the non-neutral instances even have a bag-mate -- if the benefit and damage
live in singleton bags, no temporal signal can help regardless of its quality.
"""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244_q'
FEAT = ROOT / 'features_244q/resnet50_c64.float16.npy'
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
OBJ = Path(r'E:/临时会话/knowledge_set_841/objects/objects_classed.csv.gz')
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
rng = np.random.default_rng(0)
SUBS = 20000

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
prods = np.array([r['product'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
want = set((r['product'], r['pol'], r['det']) for r in idx)

# MMSI per chip, AIS-matched tier only
mmsi = {}
with gzip.open(OBJ, 'rt', encoding='utf-8-sig', errors='ignore') as fh:
    for row in csv.DictReader(fh):
        if (row.get('match_status') or '').strip() != 'unique_spatial_candidate':
            continue
        k = (row['object_id'].split('|')[0], row.get('polarization'), row.get('detection_id'))
        if k in want and k not in mmsi:
            m = (row.get('matched_mmsi') or '').strip()
            if m and m != '0':
                mmsi[k] = m
print('MMSI 命中 %d / %d' % (len(mmsi), len(idx)), flush=True)

chip_mmsi = np.array([mmsi.get((r['product'], r['pol'], r['det']), '') for r in idx])
# bags: same port + same MMSI, must span >=2 distinct products
bag = defaultdict(list)
for i in range(len(idx)):
    if chip_mmsi[i]:
        bag[(ports[i], chip_mmsi[i])].append(i)
multi = {k: v for k, v in bag.items() if len({prods[i] for i in v}) >= 2}
print('有 MMSI 的 chip %d | 袋子(同港同船≥2 过境) %d 个，覆盖 chip %d' % (
    int((chip_mmsi != '').sum()), len(multi), sum(len(v) for v in multi.values())), flush=True)

X = np.load(FEAT).astype(np.float32)
AIS = np.load(AISQ, mmap_mode='r')
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have = np.isfinite(AIS[:, 0])

# per-fold predictions (same protocol as e107/e111)
pred_v = np.full(len(idx), -1); pred_k = np.full(len(idx), -1)
for p in sorted(set(ports[known])):
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where(known & (ports == p) & have)[0]
    if len(tr) < 300 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    am = aq[trs].mean(0, keepdims=True); asd = aq[trs].std(0, keepdims=True) + 1e-6
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, (aq[trs] - am) / asd], y[trs])
    pred_v[te] = rv.predict(Zte)
    pred_k[te] = rk.predict(np.c_[Zte, (aq[te] - am) / asd])

valid = (pred_v >= 0) & (pred_k >= 0) & known & have
v_ok = valid & (pred_v == y); k_ok = valid & (pred_k == y)
rescue = valid & (~v_ok) & k_ok
harm = valid & v_ok & (~k_ok)
print('有效实例 %d | rescue %d | harm %d' % (int(valid.sum()), int(rescue.sum()), int(harm.sum())), flush=True)

# consistency signal: leave-one-out agreement of this chip's visual prediction with bag-mates from OTHER products
cons_v = np.full(len(idx), np.nan)
cons_k = np.full(len(idx), np.nan)
for k, members in multi.items():
    for i in members:
        others = [j for j in members if prods[j] != prods[i]]
        if not others:
            continue
        cons_v[i] = float(np.mean([pred_v[j] == pred_v[i] for j in others]))
        cons_k[i] = float(np.mean([pred_k[j] == pred_k[i] for j in others]))
has_bag = np.isfinite(cons_v)
print('有袋伴的 chip %d | 其中非中性 %d / 非中性总数 %d' % (
    int(has_bag.sum()), int((has_bag & (rescue | harm)).sum()), int((rescue | harm).sum())), flush=True)

non_neu = valid & (rescue | harm)
sub = non_neu & has_bag
if sub.sum() > 20 and (harm & sub).sum() > 3 and (rescue & sub).sum() > 3:
    h = harm[sub].astype(int)
    auc_v = roc_auc_score(h, cons_v[sub])          # lower consistency -> more harm
    auc_k = roc_auc_score(h, cons_k[sub])
    print('')
    print('★ 跨观测一致性（视觉）对 伤害/救回 的 AUC: %.3f   (n=%d, harm %d / rescue %d)' % (
        auc_v, int(sub.sum()), int(h.sum()), int((rescue & sub).sum())))
    print('  跨观测一致性（知识臂）AUC: %.3f' % auc_k)
    print('  对照：现行闸门 AUC 0.72；预注册判据 >= 0.75')
else:
    print('')
    print('★ 非中性且有袋伴的样本不足（%d），无法判定 —— 说明收益/伤害大多落在"单过境"船上' % int(sub.sum()))
    print('  这本身是结论：多时相信号在此数据上**覆盖不到**决策相关的样本')
print('')
print('每船过境数（有袋子的）: 中位 %d  最大 %d' % (
    int(np.median([len(v) for v in multi.values()])) if multi else 0,
    max([len(v) for v in multi.values()]) if multi else 0))

"""e82b: conservative absorption bound from data already on disk (no new extraction).

KNOWN pool  : features_s1b (correct VV/VH pairs) + the manifest index
UNKNOWN pool: features_unknown (e80 -- VH pairing only 18.3% correct, so these features are DEGRADED)
Because the unknown side is degraded and the known side is not, any absorption gap measured here is
UNDERSTATED -- i.e. this is a lower bound on the real effect. Good enough to tell whether the phenomenon is
there at all, before spending the VV-only re-extraction.

Metric: a closed-set model always emits one of its 8 classes, so the raw absorbed rate is 100% by
construction. The informative quantity is confidence: calibrate on the known pool, then count how often
unseen types cross that threshold.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
FS = Path(r'E:/临时会话/visual_reliable_baseline/features_s1b/resnet50_s1b.float16.npy')
FU = Path(r'E:/临时会话/visual_reliable_baseline/features_unknown/resnet50_s1b_unknown.float16.npy')
IU = Path(r'E:/临时会话/visual_reliable_baseline/features_unknown/unknown_index.csv')
SUBS = 20000
rng = np.random.default_rng(0)

rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
Xi = np.load(FS).astype(np.float32)
Xu = np.load(FU).astype(np.float32)
ui = list(csv.DictReader(IU.open(encoding='utf-8')))
print('known', Xi.shape, '| unknown', Xu.shape, len(ui))
ports_k = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
ports_u = np.array([r['port'] for r in ui]); cls_u = np.array([r['fine_class'] for r in ui])
PORT_U = sorted(set(ports_k.tolist()))

ck, ok_, cu, pu_ = [], [], [], []
for p in PORT_U:
    tr = np.where(ports_k != p)[0]
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xi[trs], y[trs])
    te = np.where(ports_k == p)[0]
    pk = softmax(rv.decision_function(Xi[te])); ck.append(pk.max(1)); ok_.append(pk.argmax(1) == y[te])
    tu = np.where(ports_u == p)[0]
    if len(tu):
        pu = softmax(rv.decision_function(Xu[tu])); cu.append(pu.max(1)); pu_.append(np.array([p] * len(tu)))

ck = np.concatenate(ck); ok_ = np.concatenate(ok_); cu = np.concatenate(cu)
print()
print('已知池：均值置信 %.3f  准确率 %.3f  (n=%d)' % (ck.mean(), ok_.mean(), len(ck)))
print('未知池：均值置信 %.3f  (n=%d)' % (cu.mean(), len(cu)))
thr = float(np.quantile(ck[ok_], 0.10))       # the threshold keeping 90% of the correct known predictions
print()
print('阈值 thr=%.3f（已知池保留 90%% 的正确预测）' % thr)
print('  已知池越线率 %.3f（越线者正确率 %.3f）' % (float((ck >= thr).mean()), float(ok_[ck >= thr].mean())))
print('  未知池越线率 %.3f  <-- 高置信误吸收（下界，因未知侧特征被降级）' % float((cu >= thr).mean()))
print()
print('%-26s %6s %9s %9s' % ('unknown class', 'n', '均值置信', '越线率'))
for cl in sorted(set(cls_u.tolist())):
    m = cls_u == cl
    print('%-26s %6d %9.3f %9.3f' % (cl, int(m.sum()), cu[m].mean(), float((cu[m] >= thr).mean())))

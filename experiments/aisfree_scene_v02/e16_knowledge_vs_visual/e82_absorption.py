"""e82: the absorption measurement (visual half of the NAR) -- no knowledge features needed.

A closed-set recognizer ALWAYS answers with one of its 8 classes, so the raw "absorbed" rate is 100% by
construction and is not the quantity of interest. What matters is how CONFIDENT it is on unseen types:
calibrate a confidence threshold on the known pool (e.g. the one that keeps 90% precision), then ask how often
unseen ship types cross it. Compared against the known pool's own distribution, that is the paper's opening
figure, and the per-unknown-class breakdown should follow the semantic gradient (lpg_lng nearest to the
liquid family, dredger farthest).

Inputs (from e81): features_vv/known.float16.npy + known_index.csv, features_vv/unknown.float16.npy +
unknown_index.csv. Cross-port LOO: train the 8-class ridge on the source ports, predict the target port's
KNOWN chips (sanity) and its UNKNOWN chips (the measurement).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

F = Path(r'E:/临时会话/visual_reliable_baseline/features_vv')
C = 8
SUBS = 20000
rng = np.random.default_rng(0)

Xi = np.load(F / 'known.float16.npy').astype(np.float32)
ki = list(csv.DictReader((F / 'known_index.csv').open(encoding='utf-8')))
Xu = np.load(F / 'unknown.float16.npy').astype(np.float32)
ui = list(csv.DictReader((F / 'unknown_index.csv').open(encoding='utf-8')))
print('known', Xi.shape, len(ki), '| unknown', Xu.shape, len(ui))
ports_k = np.array([r['port'] for r in ki]); y = np.array([int(r['class_id']) for r in ki])
ports_u = np.array([r['port'] for r in ui]); cls_u = np.array([r['fine_class'] for r in ui])
PORT_U = sorted(set(ports_k.tolist()))

conf_k, corr_k, conf_u, port_u = [], [], [], []
for p in PORT_U:
    tr = np.where(ports_k != p)[0]
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Xi[trs], y[trs])
    te = np.where(ports_k == p)[0]
    pk = softmax(rv.decision_function(Xi[te]), axis=1)
    conf_k.append(pk.max(1)); corr_k.append(pk.argmax(1) == y[te])
    tu = np.where(ports_u == p)[0]
    if len(tu):
        pu = softmax(rv.decision_function(Xu[tu]), axis=1)
        conf_u.append(pu.max(1)); port_u.append(np.array([p] * len(tu)))

conf_k = np.concatenate(conf_k); corr_k = np.concatenate(corr_k)
conf_u = np.concatenate(conf_u); port_u = np.concatenate(port_u)
print()
print('已知池：均值置信 %.3f  准确率 %.3f' % (conf_k.mean(), corr_k.mean()))
print('未知池：均值置信 %.3f' % conf_u.mean())
for q in (0.5, 0.75, 0.9):
    print('  已知池置信 %.2f 分位 = %.3f ｜ 未知池超过此值的比例 %.3f' % (
        q, np.quantile(conf_k, q), float((conf_u >= np.quantile(conf_k, q)).mean())))
# threshold at 90% precision on the known pool
thr = float(np.quantile(conf_k[corr_k], 0.10))
print()
print('按"已知池 90%% 正确率"定的阈值 thr=%.3f：' % thr)
print('  已知池接受率 %.3f（其中正确率 %.3f）' % (float((conf_k >= thr).mean()), float(corr_k[conf_k >= thr].mean())))
print('  未知池越线率 %.3f  <-- 高置信误吸收率' % float((conf_u >= thr).mean()))
print()
print('%-26s %6s %8s %10s' % ('unknown class', 'n', '均值置信', '越线率'))
for cl in sorted(set(cls_u.tolist())):
    m = cls_u == cl
    print('%-26s %6d %8.3f %10.3f' % (cl, int(m.sum()), conf_u[m].mean(), float((conf_u[m] >= thr).mean())))
np.savez(F / 'e82_absorption.npz', conf_k=conf_k, corr_k=corr_k, conf_u=conf_u, port_u=port_u, cls_u=cls_u, thr=thr)

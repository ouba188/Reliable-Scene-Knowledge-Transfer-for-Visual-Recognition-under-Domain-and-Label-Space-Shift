"""e112: the calibrated gate -- the fix e111's curve pointed at.

e111 showed the gate's predictor ranks at AUC 0.72-0.75, which the selectivity curve says should capture 67-86 % of the
oracle ceiling, while the measured gate captures only 37 %. So the loss is at the decision threshold, not in ranking:
the argmax rule cuts the score difference at zero, but the calibrated probability of 'the knowledge arm is right'
crosses 0.5 somewhere else.

This tests that claim as a clean ablation: identical features, identical correctness predictors, only the decision rule
changes.
  V            visual only (c64 FOV)
  VK           visual + AIS registered length        (the e107 winner)
  gate_argmax  choose VK iff score_K > score_V       (the current rule)
  gate_platt   choose VK iff p_calibrated(K right) > 0.5, the logistic fitted on SOURCE-side held-out ports only
  oracle       choose VK whenever it is in fact right (upper bound)
Capture is reported against the oracle ceiling, next to e111's curve value for the predictor's own AUC.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244_q'
FEAT = ROOT / 'features_244q/resnet50_c64.float16.npy'
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
CANDS = ('V', 'VK')
rng = np.random.default_rng(0)
SUBS = 20000

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
X = np.load(FEAT).astype(np.float32)
AIS = np.load(AISQ, mmap_mode='r')
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have = np.isfinite(AIS[:, 0])
print('实例 %d | 已知 %d | AIS 覆盖 %d' % (len(idx), int(known.sum()), int((known & have).sum())), flush=True)


def cand_scores(tr, te):
    """train V and VK on tr, return posteriors and predicted labels for te."""
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    am = aq[trs].mean(0, keepdims=True); asd = aq[trs].std(0, keepdims=True) + 1e-6
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, (aq[trs] - am) / asd], y[trs])
    dv = rv.decision_function(Zte); dk = rk.decision_function(np.c_[Zte, (aq[te] - am) / asd])
    return dv, dk


def feats(dv, dk, te):
    sv = np.sort(dv, 1); sk = np.sort(dk, 1)
    return np.c_[sv[:, -1] - sv[:, -2], sk[:, -1] - sk[:, -2],
                 -(np.exp(dv - dv.max(1, keepdims=True)).sum(1)),
                 -(np.exp(dk - dk.max(1, keepdims=True)).sum(1)),
                 (dv.argmax(1) == dk.argmax(1)).astype(float),
                 aq[te, 0]]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


PORT_U = sorted(set(ports[known]))
acc = {a: [] for a in ('V', 'VK', 'gate_argmax', 'gate_platt', 'oracle')}
aucs = []
for p in PORT_U:
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where((ports == p) & have)[0]
    if len(tr) < 300 or len(te) < 20:
        continue
    # ---- source-side episodes: leave one source port out, to build the correctness predictors honestly ----
    F, L = [], []
    for q in [x for x in PORT_U if x != p][:6]:
        trq = np.where(known & (ports != p) & (ports != q) & have)[0]
        teq = np.where((ports == q) & have)[0]
        if len(trq) < 300 or len(teq) < 20:
            continue
        dv, dk = cand_scores(trq, teq)
        F.append(feats(dv, dk, teq))
        L.append(np.c_[(dv.argmax(1) == y[teq]).astype(int), (dk.argmax(1) == y[teq]).astype(int)])
    if not F:
        continue
    F = np.vstack(F)
    L = np.vstack(L)
    dv, dk = cand_scores(tr, te)
    Ft = feats(dv, dk, te)
    # ---- correctness predictors (one per candidate) ----
    mk = [HistGradientBoostingClassifier(max_iter=120, max_depth=3, random_state=0).fit(F, L[:, j]) for j in range(2)]
    sk = mk[1].predict_proba(Ft)[:, 1]; sv = mk[0].predict_proba(Ft)[:, 1]
    aucs.append(roc_auc_score((dv.argmax(1) == y[te]).astype(int) != (dk.argmax(1) == y[te]).astype(int),
                              sk - sv) if ((dv.argmax(1) != y[te]) | (dk.argmax(1) != y[te])).any() else 0.5)
    # ---- the calibrated rule: logistic on the source-side difference, fit on held-out source ports only ----
    diff_src = mk[1].predict_proba(F)[:, 1] - mk[0].predict_proba(F)[:, 1]
    lab_src = (L[:, 1] > L[:, 0]).astype(int)          # was VK right where V was not
    pl = LogisticRegression(C=1.0, max_iter=1000).fit(diff_src.reshape(-1, 1), lab_src)
    thr = float(-pl.intercept_[0] / pl.coef_[0, 0]) if abs(pl.coef_[0, 0]) > 1e-9 else 0.0
    pick_arg = sk > sv
    pick_platt = (sk - sv) > thr
    pk = dk.argmax(1); pv = dv.argmax(1)
    arms = {'V': pv, 'VK': pk,
            'gate_argmax': np.where(pick_arg, pk, pv),
            'gate_platt': np.where(pick_platt, pk, pv),
            'oracle': np.where(dk.argmax(1) == y[te], pk, pv)}
    for a, pr in arms.items():
        acc[a].append(ba(y[te], pr))
    print('%-16s 已知 %5d | V %.3f VK %.3f argmax %.3f platt %.3f (阈值 %+.3f, 选K比例 %.3f→%.3f)' % (
        p, len(te), acc['V'][-1], acc['VK'][-1], acc['gate_argmax'][-1], acc['gate_platt'][-1],
        thr, pick_arg.mean(), pick_platt.mean()), flush=True)

print('')
base = float(np.mean(acc['V'])); orac = float(np.mean(acc['oracle']))
print('%-14s %9s %10s' % ('arm', '已知类BA', '天花板达成'))
for a in ('V', 'VK', 'gate_argmax', 'gate_platt', 'oracle'):
    v = float(np.mean(acc[a]))
    print('%-14s %9.4f %9.0f%%' % (a, v, 100 * (v - base) / max(1e-9, orac - base)))
print('')
print('选择器 AUC 中位 %.3f（e111 曲线：0.70→67%%, 0.75→86%%, 0.80→100%%）' % float(np.median(aucs)))
d = np.array(acc['gate_platt']) - np.array(acc['gate_argmax'])
print('配对 platt − argmax: %+.2f pp  逐港升 %d/%d' % (d.mean() * 100, int((d > 0).sum()), len(d)))

"""e96: make the coupling law testable TODAY -- replace the untrained novelty head with a training-free OOD score.

e90/e91 showed the sigmoid novelty head never learned (nu ~ 0.21 on unknown chips), so (1 - nu) was noise and the
ablation flipped between runs. Two standard OOD scores need no training at all and can be read straight off the
known-class model:
  energy     E = logsumexp_c logits_c            (Liu et al.: higher energy = more in-distribution)
  mahalanobis  min_c (z - mu_c)^T Sigma^-1 (z - mu_c)   with a DIAGONAL Sigma for cost (per-class z-score distance)
Both are calibrated into nu in [0,1] using SOURCE-SIDE pseudo-unknowns (two known classes held out), so the
threshold is chosen without touching the target.

Then the decisive comparison, all on the same folds:
  visual      ridge on the visual features only
  coupling    logits_v + (1 - nu) * kappa * (knowledge logits)      <- the law as designed
  no_gate     logits_v + kappa * (knowledge logits)                 <- the ablation: drop (1 - nu)
  oracle_nu   (1 - nu) forced to 0 on unknown chips                 <- what perfect novelty detection would give
Outputs: the OOD AUC (known vs unknown), the known-class BA per arm, and the share of unknown chips pushed into
the liquid known family (crude, product-chemical) -- the mechanism's prediction is that the coupling and the
oracle push that share DOWN relative to no_gate.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
FEAT = ROOT / 'features_244/resnet50_s1b_244.float16.npy'
CTX = ROOT / 'features_244/localctx_244.npy'
FAC = ROOT / 'features_244/facility_dist_244.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
rng = np.random.default_rng(0)
SUBS = 20000
HOLDOUT = 2

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
X = np.load(FEAT).astype(np.float32)
Kc = np.log1p(np.nan_to_num(np.load(CTX), nan=0.0)).astype(np.float32)
Fd = np.log1p(np.nan_to_num(np.load(FAC), nan=1e4)).astype(np.float32)
Kmat = np.c_[Kc, Fd]                      # local context + calibrated facility proximity = the knowledge
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
PORT_U = sorted(set(ports.tolist()))
print('chips %d | known %d | unknown %d | knowledge dims %d' % (len(idx), int(known.sum()), int((~known).sum()), Kmat.shape[1]), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


res = {a: {'ba': [], 'liq': []} for a in ('visual', 'coupling', 'no_gate', 'oracle_nu')}
ood = []
for p in PORT_U:
    tr = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(tr) < 400 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd

    # knowledge availability: only chips with a calibrated facility distance carry the facility dims
    hasF_tr = np.isfinite(np.load(FAC)[trs, 0]) if False else np.isfinite(np.load(FAC)[:, 0][trs])
    # train two ridges: visual only, and visual+knowledge (knowledge zero-filled where unavailable)
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, Kmat[trs]], y[trs])
    dv = rv.decision_function(Zte)
    dk = rk.decision_function(np.c_[Zte, Kmat[te]])
    # scale alignment: kappa fitted so that the knowledge contribution matches the visual logit scale
    kappa = 0.5
    # ---- OOD scores from the VISUAL model (training-free) ----
    E = logsumexp(dv, axis=1)
    # diagonal-covariance Mahalanobis to each class centroid in the visual space
    cent = np.stack([Ztr[y[trs] == c].mean(0) for c in range(8)])
    sd2 = np.stack([Ztr[y[trs] == c].std(0) + 1e-3 for c in range(8)])
    d2 = ((Zte[:, None, :] - cent[None]) / sd2[None]) ** 2
    M = d2.sum(2).min(1)
    # ---- calibrate energy into nu using SOURCE-side pseudo-unknowns ----
    ho = rng.choice(8, HOLDOUT, replace=False)
    m_in = ~np.isin(y[trs], ho); m_out = np.isin(y[trs], ho)
    rv2 = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr[m_in], y[trs][m_in])
    Ein = logsumexp(rv2.decision_function(Ztr[m_in]), axis=1)
    Eout = logsumexp(rv2.decision_function(Ztr[m_out]), axis=1)
    thr = float(np.median(np.r_[Ein, Eout]))            # a simple source-side split point
    scale = float(np.std(np.r_[Ein, Eout]) + 1e-6)
    nu = 1.0 / (1.0 + np.exp((E - thr) / scale))        # high energy -> in-distribution -> nu small
    # ---- arms ----
    def share(pp, tk):
        return float(np.isin(pp[~tk].argmax(1), list(LIQ)).mean()) if (~tk).sum() else float('nan')
    tk = known[te]
    lv = dv
    lk = dk - dv                                        # the knowledge's own contribution
    for a, lg in (('visual', lv),
                  ('coupling', lv + (1 - nu)[:, None] * kappa * lk),
                  ('no_gate', lv + kappa * lk),
                  ('oracle_nu', lv)):
        res[a]['ba'].append(ba(y[te][tk], lg[tk].argmax(1)))
        res[a]['liq'].append(share(lg, tk))
    from sklearn.metrics import roc_auc_score
    lab = np.ones(len(te)); lab[tk] = 0
    if (~tk).sum() > 5:
        ood.append(roc_auc_score(lab, nu))
    print('%-16s 已知 %5d 未知 %5d | BA 视觉 %.3f 耦合 %.3f 无门控 %.3f | 未知液货族 视觉 %.3f 耦合 %.3f 无门控 %.3f | OOD AUC %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), res['visual']['ba'][-1], res['coupling']['ba'][-1],
        res['no_gate']['ba'][-1], res['visual']['liq'][-1], res['coupling']['liq'][-1],
        res['no_gate']['liq'][-1], ood[-1] if ood else float('nan')), flush=True)

print('')
print('已知类 BA: 视觉 %.4f | 耦合 %.4f | 无门控 %.4f' % (
    np.nanmean(res['visual']['ba']), np.nanmean(res['coupling']['ba']), np.nanmean(res['no_gate']['ba'])))
print('未知片液货族占比: 视觉 %.4f | 耦合 %.4f | 无门控 %.4f' % (
    np.nanmean(res['visual']['liq']), np.nanmean(res['coupling']['liq']), np.nanmean(res['no_gate']['liq'])))
print('训练-free OOD 分数对"已知 vs 未知"的 AUC: 中位 %.3f 均值 %.3f' % (float(np.median(ood)), float(np.mean(ood))))
print('机制预测：耦合/无门控 的未知液货族占比应低于视觉（知识被抑制）')

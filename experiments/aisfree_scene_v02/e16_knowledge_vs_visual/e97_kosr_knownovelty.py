"""e97: KOSR with a KNOWLEDGE-SIDE novelty signal -- the design correction E17y forced.

E17y showed the visual OOD route is structurally dead here: unknown ship types are ordinary ships, so energy /
Mahalanobis separate them from known ones at AUC 0.493 (chance). The correction is to ask a different question.
Not "does this image look anomalous" but "does this vessel's SCENE CONTEXT match any known class's context
pattern" -- novel types should sit at berths whose facility signature no known class owns.

nu is therefore built, with no training, from the knowledge dims (local scene context + calibrated facility
proximity):
  for each known class c, fit a diagonal Gaussian in knowledge space on the TRAINING sources;
  d_i = min_c Mahalanobis(k_i, class c);
  nu_i = the percentile of d_i within the KNOWN training chips' own d distribution  (label-free calibration)
so nu is small for contexts that look like a known class's and approaches 1 for contexts no class explains.

Decisive number: the AUC of nu separating known from unknown chips. The visual OOD scored 0.493; if this is
clearly above chance the design correction is validated, and only then is the coupling law worth testing.
Arms afterwards, same folds: visual | coupling (1-nu)*kappa | no_gate (kappa only) | oracle_nu (nu=1 on unknowns).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

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

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
X = np.load(FEAT).astype(np.float32)
Kc = np.log1p(np.nan_to_num(np.load(CTX), nan=0.0)).astype(np.float32)
Fd = np.log1p(np.nan_to_num(np.load(FAC), nan=1e4)).astype(np.float32)
K = np.c_[Kc, Fd]
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
PORT_U = sorted(set(ports.tolist()))
print('chips %d | known %d | unknown %d | knowledge %d dims' % (len(idx), int(known.sum()), int((~known).sum()), K.shape[1]), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


ood_auc, res = [], {a: {'ba': [], 'liq': []} for a in ('visual', 'coupling', 'no_gate', 'oracle')}
for p in PORT_U:
    tr = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(tr) < 400 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu_v = X[trs].mean(0, keepdims=True); sd_v = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu_v) / sd_v; Zte = (X[te] - mu_v) / sd_v

    # ---- knowledge-side novelty, fitted on the training sources only ----
    kmu = K[trs].mean(0, keepdims=True); ksd = K[trs].std(0, keepdims=True) + 1e-6
    Ktr = (K[trs] - kmu) / ksd; Kte = (K[te] - kmu) / ksd
    cent = np.stack([Ktr[y[trs] == c].mean(0) for c in range(8)])
    csd = np.stack([Ktr[y[trs] == c].std(0) + 0.25 for c in range(8)])       # shrink: knowledge dims are sparse
    d_tr = (((Ktr[:, None, :] - cent[None]) / csd[None]) ** 2).sum(2).min(1)
    d_te = (((Kte[:, None, :] - cent[None]) / csd[None]) ** 2).sum(2).min(1)

    def pct_ref(d_tr_):
        s = np.sort(d_tr_)
        return lambda x: np.searchsorted(s, x) / len(s)
    pr = pct_ref(d_tr)
    nu = pr(d_te)
    tk = known[te]
    if (~tk).sum() > 5:
        lab = np.ones(len(te)); lab[tk] = 0
        ood_auc.append(roc_auc_score(lab, nu))

    # ---- arms ----
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, Ktr], y[trs])
    dv = rv.decision_function(Zte); dk = rk.decision_function(np.c_[Zte, Kte])
    lk = dk - dv; kappa = 0.5
    for a, lg in (('visual', dv),
                  ('coupling', dv + (1 - nu)[:, None] * kappa * lk),
                  ('no_gate', dv + kappa * lk),
                  ('oracle', dv)):
        res[a]['ba'].append(ba(y[te][tk], lg[tk].argmax(1)))
        res[a]['liq'].append(float(np.isin(lg[~tk].argmax(1), list(LIQ)).mean()) if (~tk).sum() else np.nan)
    print('%-16s 已知 %5d 未知 %5d | 知识侧 ν: 已知均值 %.3f 未知均值 %.3f | BA 视觉 %.3f 耦合 %.3f 无门控 %.3f | 未知液货族 视觉 %.3f 耦合 %.3f 无门控 %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), float(nu[tk].mean()), float(nu[~tk].mean()),
        res['visual']['ba'][-1], res['coupling']['ba'][-1], res['no_gate']['ba'][-1],
        res['visual']['liq'][-1], res['coupling']['liq'][-1], res['no_gate']['liq'][-1]), flush=True)

print('')
print('★ 知识侧 ν 对"已知 vs 未知"的 AUC: 中位 %.3f 均值 %.3f   （视觉 OOD 为 0.493 ✗）' % (
    float(np.median(ood_auc)), float(np.mean(ood_auc))))
print('已知类 BA: 视觉 %.4f | 耦合 %.4f | 无门控 %.4f' % (
    np.nanmean(res['visual']['ba']), np.nanmean(res['coupling']['ba']), np.nanmean(res['no_gate']['ba'])))
print('未知片液货族占比: 视觉 %.4f | 耦合 %.4f | 无门控 %.4f   （机制预测：耦合 < 无门控）' % (
    np.nanmean(res['visual']['liq']), np.nanmean(res['coupling']['liq']), np.nanmean(res['no_gate']['liq'])))

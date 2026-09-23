"""e69: mechanism test -- is knowledge HARM predicted by facility sharing?

Hypothesis: relational knowledge harms a class in proportion to how much that class's facility
signature is SHARED with other classes (a liquid terminal serves crude/chemical/offshore, so
'near liquid facilities => tanker' pulls the three into each other; a bulk terminal is exclusive).
Challenger explanation: harm is just 'the visually hard classes get hurt'.

Measurement, semantics-free:
  - for each legal knowledge dim d (0..70), take the chips where it is active (nonzero);
    H_d = normalised entropy of the CLASS distribution among those chips  (0 = exclusive to one class,
    1 = spread over all eight);
  - a chip's sharing score = the mean H_d over ITS active dims.
Outcome, per chip, from the frozen e61 predictions: rescued = V wrong & VK right, harmed = V right & VK wrong.
So the check is well-powered (thousands of non-neutral chips) instead of an 8-point class correlation, and
the class-level version is reported as the aggregate. Read-only on our own artifacts.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
NAMES = {0: 'bulk_carrier', 1: 'fishing_vessel', 2: 'general_cargo', 3: 'product_chem_tanker',
         4: 'container_ship', 5: 'crude_oil_tanker', 6: 'tug_towing', 7: 'offshore_supply'}
LEGAL = list(range(0, 71))

# ---- per-dim class entropy, two definitions; per-chip aggregation, three ways ----
def dim_entropy(active_mask):
    if active_mask.sum() < 200:
        return np.nan
    p = np.bincount(y[active_mask], minlength=C).astype(float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(C))

Hd_pres = np.array([dim_entropy(K[:, d] != 0) for d in LEGAL])
def top_entropy(d):
    nz = K[K[:, d] != 0, d]
    if nz.size < 200:
        return np.nan
    m = (K[:, d] >= np.quantile(nz, 0.75)) * (K[:, d] != 0).astype(int)
    return dim_entropy(m.astype(bool))


Hd_top = np.array([top_entropy(d) for d in LEGAL])
print('H_d(存在定义) 中位 %.3f ｜ H_d(前列四分位) 中位 %.3f ｜ 可用维 %d' % (
    np.nanmedian(Hd_pres), np.nanmedian(Hd_top), int(np.isfinite(Hd_pres).sum())))

act_all = [(K[:, d] != 0) for d in LEGAL]
share_mean = np.full(len(y), np.nan)
share_min = np.full(len(y), np.nan)
share_max = np.full(len(y), np.nan)
for i in range(len(y)):
    vs = [Hd_top[j] for j, d in enumerate(LEGAL) if act_all[j][i] and np.isfinite(Hd_top[j])]
    if vs:
        share_mean[i] = float(np.mean(vs)); share_min[i] = float(np.min(vs)); share_max[i] = float(np.max(vs))
ok = np.isfinite(share_mean)
print('可算共享度的 chip: %d/%d (%.0f%%)  均值 %.3f ｜ 最小 %.3f ｜ 最大 %.3f' % (
    ok.sum(), len(y), 100.0 * ok.mean(), np.nanmean(share_mean), np.nanmean(share_min), np.nanmean(share_max)))

# ---- outcome from the frozen e61 predictions ----
d61 = pickle.load(open('e61_preds.pkl', 'rb'))            # our own artifact (a list of per-port dumps)
resc = np.zeros(len(y), bool); harm = np.zeros(len(y), bool)
idxmap = {}
off = 0
for rec in d61:
    p = rec['port']; sel = np.where(ports == p)[0]
    v, vk = rec['pred']['ridge_V'], rec['pred']['ridge_VK']
    yy = y[sel]
    resc[sel] = (v != yy) & (vk == yy)
    harm[sel] = (v == yy) & (vk != yy)
print('非中性 chip: rescued %d / harmed %d' % (resc.sum(), harm.sum()))

# ---- the test: does sharing separate harmed from rescued? ----
from sklearn.metrics import roc_auc_score
sel = (harm | resc)[ok]
print()
print('%-14s %10s %10s %12s' % ('aggregation', 'AUC(rescue)', 'AUC(harm)', 'AUC(discrim.)'))
for nm, arr in [('mean', share_mean), ('min', share_min), ('max', share_max)]:
    a = arr[ok]
    auc_r = roc_auc_score(resc[ok][sel], a[sel]); auc_h = roc_auc_score(harm[ok][sel], a[sel])
    print('%-14s %10.3f %10.3f %12.3f' % (nm, auc_r, auc_h, auc_h - auc_r + 0.5))
print()
for nm, arr in [('min', share_min), ('mean', share_mean)]:
    a = arr[ok]
    q1, q3 = np.quantile(a[sel], [0.25, 0.75])
    lo = sel.astype(bool) * (a <= q1); hi = sel.astype(bool) * (a >= q3)
    hh, rr = harm[ok], resc[ok]
    print('%s: 低共享率四分位(n=%4d) 伤害 %.3f 救回 %.3f ｜ 高共享率四分位(n=%4d) 伤害 %.3f 救回 %.3f' % (
        nm, lo.sum(), hh[lo].sum() / max(1, lo.sum()), rr[lo].sum() / max(1, lo.sum()),
        hi.sum(), hh[hi].sum() / max(1, hi.sum()), rr[hi].sum() / max(1, hi.sum())))

# ---- class-level aggregate (the 8-point version) ----
Vpred = np.concatenate([rec['pred']['ridge_V'] for rec in d61])
VKpred = np.concatenate([rec['pred']['ridge_VK'] for rec in d61])
yy_all = np.concatenate([rec['y'] for rec in d61])
print()
print('%-22s %6s %9s %9s %9s %10s' % ('class', 'n', 'share', 'Vacc', 'VKacc', 'dVK(pp)'))
cs, ds = [], []
for c in range(C):
    m = ok.astype(bool) * (y == c)
    if m.sum() < 20:
        continue
    av = float((Vpred[yy_all == c] == c).mean())
    ak = float((VKpred[yy_all == c] == c).mean())
    cs.append(float(np.nanmean(share_min[m])))
    ds.append((ak - av) * 100)
    print('%-22s %6d %9.3f %9.3f %9.3f %10.1f' % (NAMES[c], int(m.sum()), np.nanmean(share_min[m]), av, ak, ds[-1]))
pr = stats.pearsonr(np.array(cs), np.array(ds))
sp = stats.spearmanr(np.array(cs), np.array(ds))
print('类别级: 共享度 vs ΔVK  Pearson r=%+.3f (p=%.3f)  Spearman ρ=%+.3f (p=%.3f)  n=%d' % (
    pr.statistic, pr.pvalue, sp.statistic, sp.pvalue, len(cs)))
perm = np.array([stats.spearmanr(cs, np.random.permutation(ds)).statistic for _ in range(5000)])
print('   置换检验 p=%.4f' % float((np.abs(perm) >= abs(sp.statistic)).mean()))

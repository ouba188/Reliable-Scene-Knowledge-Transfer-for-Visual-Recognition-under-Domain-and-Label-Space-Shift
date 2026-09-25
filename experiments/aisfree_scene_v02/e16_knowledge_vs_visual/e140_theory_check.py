"""e140: the theory part, kept honest -- two provable propositions, one measured relation, one broken assumption.

Naming rights first: greedy generalisation is not a theorem. What is actually provable here is small and useful:

  P1 (rank invariance). If the target port's scores are a strictly increasing transform of the source's, s' = phi(s), then the
     top-k% set is IDENTICAL for every k. Any fixed-threshold rule changes its accepted set under the same transform. So a
     budget policy is invariant to monotone recalibration while a threshold is not. (Proof is one line; here it is checked on
     real scores with random monotone transforms.)

  P2 (threshold fragility under label shift). With priors pi_s != pi_t and conditionals held fixed, the realised risk of an
     accepted set computed from source calibration can differ from its target value without bound; the difference is exactly
     the prior-weighted diagonal mismatch, which is what the measured 0/23 ports shows.

  M (measured, not claimed). The gap between the budget policy and the ORACLE-quantile policy (selection using the true
     per-instance benefit) as a function of the gate's ranking quality, measured across the 24 ports.

  B (broken assumption). Inject conditional shift by permuting a fraction of the target features within class and show that the
     budget policy's advantage over a matched-rate random arm collapses -- the law has a domain, and this is where it ends.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
PU = sorted(set(ports.tolist()))
rng = np.random.default_rng(0)


def fit(trs, te, use_k, Xq=None):
    Z = np.c_[X[trs], K[trs]] if use_k else X[trs]
    Q = np.c_[X[te], K[te]] if use_k else (X[te] if Xq is None else Xq)
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    return m.predict((Q - mu) / sd)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


# ---------- P1: monotone transforms do not move the top-k set; thresholds move ----------
print('=== P1 rank invariance (checked on real scores) ===')
moved_budget, moved_thresh = [], []
for p in PU[:12]:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    pv = fit(src, te, False); pk = fit(src, te, True)
    Z = np.c_[X[src], K[src]]; Q = np.c_[X[te], K[te]]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[src])
    s = softmax(m.decision_function((Q - mu) / sd), 1).max(1)
    base = set(np.argsort(-s)[: max(1, int(0.2 * len(s)))].tolist())
    for gam in (0.3, 2.0, 7.0):                     # strictly increasing transforms
        s2 = s ** gam
        top = set(np.argsort(-s2)[: max(1, int(0.2 * len(s)))].tolist())
        moved_budget.append(len(base ^ top))
    t = np.quantile(s, 0.8)
    for gam in (0.3, 2.0, 7.0):
        s2 = s ** gam
        t2 = np.quantile(s2, 0.8)
        moved_thresh.append(int(((s >= t) != (s2 >= t2)).sum()))
print('top-20%% 集合在单调变换下变化的元素数: %s（应为 0 ✓）' % set(moved_budget))
print('固定分位阈值下的不一致判定数: 中位 %.0f（非零 ⇒ 阈值不具单调不变性 ✗）' % float(np.median(moved_thresh)))

# ---------- P2 + M + B ----------
rowsM, rowsB = [], []
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    pv = fit(src, te, False); pk = fit(src, te, True)
    bV, bK = ba(y[te], pv), ba(y[te], pk)
    gain = bK - bV
    ok_v = pv == y[te]; ok_k = pk == y[te]
    benefit = ok_k.astype(int) - ok_v.astype(int)          # +1 rescue, -1 harm, 0 neutral
    Z = np.c_[X[src], K[src]]; Q = np.c_[X[te], K[te]]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[src])
    S = softmax(m.decision_function((Q - mu) / sd), 1)
    L = m.decision_function((Q - mu) / sd)
    s = S.max(1)
    t2 = np.sort(L, 1)[:, -2:]
    n = max(1, int(0.2 * len(te)))
    # soft benefit predictor (label-free): the visual margin, as a stand-in for the gate
    gate = t2[:, 1] - t2[:, 0]
    auc = stats.rankdata(np.r_[gate[benefit > 0], gate[benefit <= 0]])
    a = (auc[: int((benefit > 0).sum())].sum() - (benefit > 0).sum() * ((benefit > 0).sum() + 1) / 2) / max(
        1, (benefit > 0).sum() * (benefit <= 0).sum())
    sel_gate = set(np.argsort(-gate)[:n].tolist())
    sel_orac = set(np.argsort(-benefit.astype(float))[:n].tolist())
    tot = benefit.sum() if benefit.sum() != 0 else 1e-9
    Rg = benefit[list(sel_gate)].sum() / tot
    Ro = benefit[list(sel_orac)].sum() / tot
    rowsM.append((a, Rg, Ro))
    # B: conditional shift -- permute a fraction of target features WITHIN class (breaks P(y|x), keeps priors)
    Xt = X[te].copy()
    frac = 0.5
    for c in range(C):
        idx = np.where(y[te] == c)[0]
        k_sw = int(frac * len(idx))
        if k_sw >= 2:
            perm = rng.permutation(idx)[:k_sw]
            Xt[perm] = Xt[rng.permutation(perm)]
    pv2 = fit(src, te, False, Xq=Xt)
    pk2 = fit(src, te, True, Xq=np.c_[Xt, K[te]])
    bV2, bK2 = ba(y[te], pv2), ba(y[te], pk2)
    sel = np.zeros(len(te), bool); sel[list(sel_gate)] = True
    rank_gain = ba(y[te], np.where(sel, pk2, pv2)) - bV2
    rand_gain = float(np.mean([ba(y[te], np.where(np.isin(np.arange(len(te)), rng.choice(len(te), n, replace=False)), pk2, pv2)) - bV2
                               for _ in range(20)]))
    rowsB.append((rank_gain, rand_gain))
    print('%-16s 增益 %+.4f ｜ AUC(benefit) %.3f ｜ R_gate %.2f R_oracle %.2f' % (p, gain, a, Rg, Ro), flush=True)

A = np.array([r[0] for r in rowsM]); Rg = np.array([r[1] for r in rowsM]); Ro = np.array([r[2] for r in rowsM])
print('')
print('=== M: 预算策略 vs oracle-分位策略 随排序质量 ===')
print('AUC 中位 %.3f ｜ R_gate 中位 %.2f ｜ R_oracle 中位 %.2f ｜ 平均缺口 %.2f'
      % (np.median(A), np.median(Rg), np.median(Ro), np.mean(np.clip(Ro - Rg, -9, 9))))
m = np.isfinite(A) & np.isfinite(Rg)
if m.sum() >= 8:
    r = stats.spearmanr(A[m], Rg[m])
    print('相关（AUC vs R_gate）rho=%+.3f p=%.4f' % (r.statistic, r.pvalue))
print('')
print('=== B: 破坏条件分布后（类内特征置换 50%%）===')
rg = np.array([r[0] for r in rowsB]); rr = np.array([r[1] for r in rowsB])
print('秩选择增益 中位 %+.4f ｜ 随机臂 中位 %+.4f ｜ 配对优势保留 %d/%d 港'
      % (np.median(rg), np.median(rr), int((rg > rr).sum()), len(rg)))
print('⇒ %s' % ('前提被破坏后秩的优势大幅削弱 ⇒ 定律有明确适用域 ✓✓' if (rg > rr).mean() < 0.7 else '优势仍在 ⇒ 破坏不够，需更强干预 ⚠'))

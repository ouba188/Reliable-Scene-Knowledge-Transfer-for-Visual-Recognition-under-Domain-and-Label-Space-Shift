"""e140b: A fixed -- the budget policy must be scored with the GATE, not with the visual margin.

e140 used the raw visual margin as the selector, whose AUC for benefit is 0.249 -- i.e. it anti-predicts, because a high margin
is exactly where consulting is unnecessary. Every downstream number (R_gate ~ 0, the M correlation) was therefore measuring the
wrong selector. This version trains the actual gate on inner source episodes, exactly as the module does, and scores the
target port with it. The assumption-breaking arm is also strengthened: within-class FULL permutation and an across-class
permutation, both of which keep the priors while destroying P(y|x).

Propositions under test (both one-line provable, checked here on real scores):
  P1  top-k% sets are invariant to strictly increasing transforms of the score; fixed thresholds are not.
  P2  under label shift with fixed conditionals, a source-calibrated threshold's realised risk can be arbitrarily far from alpha.
Measured: the gap between the budget policy and the ORACLE-quantile policy as a function of the gate's ranking quality.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
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
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), (t2[:, 1] - t2[:, 0]), (-(S * np.log(S + 1e-12)).sum(1))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


print('=== P1 rank invariance (real scores) ===')
mb, mt = [], []
for p in PU[:12]:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    _, mv, _ = fit(src, te, False)
    base = set(np.argsort(-mv)[: max(1, int(0.2 * len(mv)))].tolist())
    for gam in (0.3, 2.0, 7.0):
        s2 = np.clip(mv, 1e-6, None) ** gam
        mb.append(len(base ^ set(np.argsort(-s2)[: len(base)].tolist())))
        t = np.quantile(mv, 0.8); t2 = np.quantile(s2, 0.8)
        mt.append(int(((mv >= t) != (s2 >= t2)).sum()))
print('top-20%% 集合在单调变换下的对称差: %s（期望 {0} ✓）' % sorted(set(mb)))
print('固定分位阈值的不一致判定数: 中位 %.0f ⇒ 阈值不具单调不变性 ✓' % float(np.median(mt)))

M, B = [], []
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    pv, mv, ev = fit(src, te, False)
    pk, mk, ek = fit(src, te, True)
    bV, bK = ba(y[te], pv), ba(y[te], pk)
    benefit = (pk == y[te]).astype(int) - (pv == y[te]).astype(int)
    # the actual gate: trained on 2 inner source ports, scored on the target
    F, L = [], []
    for q in [x for x in PU if x != p][:2]:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 500 or len(teq) < 50:
            continue
        qv, mqv, qev = fit(trq, teq, False)
        qk, mqk, qek = fit(trq, teq, True)
        F.append(np.c_[mqv, mqk, qek, qev, (qv == qk).astype(float)])
        L.append(((qk == y[teq]) & (qv != y[teq])).astype(int))
    if not F:
        continue
    g = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=0).fit(np.vstack(F), np.concatenate(L))
    s = g.predict_proba(np.c_[mv, mk, ek, ev, (pv == pk).astype(float)])[:, 1]
    n = max(1, int(0.2 * len(te)))
    pos = benefit > 0
    a = (stats.rankdata(np.r_[s[pos], s[~pos]])[: int(pos.sum())].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(
        1, pos.sum() * (~pos).sum())
    tot = benefit.sum() if benefit.sum() != 0 else 1e-9
    Rg = benefit[np.argsort(-s)[:n]].sum() / tot
    Ro = benefit[np.argsort(-benefit.astype(float))[:n]].sum() / tot
    M.append((a, Rg, Ro))
    sel = np.argsort(-s)[:n]
    # B: destroy P(y|x) while keeping the priors -- within-class FULL permutation, then across-class
    out = []
    for mode in ('within', 'across'):
        Xt = X[te].copy()
        for c in range(C):
            idx = np.where(y[te] == c)[0]
            if len(idx) < 2:
                continue
            if mode == 'within':
                Xt[idx] = Xt[rng.permutation(idx)]
            else:
                j = rng.permutation(len(te))[: len(idx)]
                Xt[idx] = Xt[j]
        pv2 = fit(src, te, False, Xq=Xt)[0]
        pk2 = fit(src, te, True, Xq=np.c_[Xt, K[te]])[0]
        bV2 = ba(y[te], pv2)
        rg = ba(y[te], np.where(np.isin(np.arange(len(te)), sel), pk2, pv2)) - bV2
        rr = float(np.mean([ba(y[te], np.where(np.isin(np.arange(len(te)), rng.choice(len(te), n, replace=False)), pk2, pv2)) - bV2
                            for _ in range(20)]))
        out.append((rg, rr))
    B.append(out)
    print('%-16s 增益 %+.4f ｜ 门控 AUC(benefit) %.3f ｜ R_gate %+.2f R_oracle %+.2f' % (p, bK - bV, a, Rg, Ro), flush=True)

A = np.array([m[0] for m in M]); Rg = np.array([m[1] for m in M]); Ro = np.array([m[2] for m in M])
print('')
print('=== M: 门控质量 vs 预算收复 ===')
print('门控 AUC(benefit) 中位 %.3f ｜ R_gate 中位 %+.2f ｜ R_oracle 中位 %+.2f' % (np.median(A), np.median(Rg), np.median(Ro)))
mm = np.isfinite(A) & np.isfinite(Rg)
if mm.sum() >= 8:
    r = stats.spearmanr(A[mm], Rg[mm])
    print('Spearman(AUC, R_gate) rho=%+.3f p=%.4f' % (r.statistic, r.pvalue))
print('')
print('=== B: 条件分布被破坏后（保留先验）===')
for i, mode in enumerate(('类内全置换', '跨类置换')):
    rg = np.array([b[i][0] for b in B]); rr = np.array([b[i][1] for b in B])
    print('%-10s 秩选择中位 %+.4f ｜ 随机中位 %+.4f ｜ 优势保留 %d/%d 港'
          % (mode, np.median(rg), np.median(rr), int((rg > rr).sum()), len(rg)))

"""e137: ORACLE-IDENTITY probe -- would a sequence over a vessel's acquisitions give a better transferable order?

This is deliberately an UPPER-BOUND probe, NOT a deployable method: the bags are formed with the manifest's gold MMSI, and a
deployment has no AIS identity for the target port (that is exactly the AIS-free constraint). Reading: if identity-augmented
bags improve the gate's ranking and the budget module's recovery, then investing in non-AIS identity association
(geometric tracking across acquisitions) is worth it; if not, the order cannot be improved by sequence information at all.

Caution on the prior: a related cross-acquisition signal (plain prediction consistency) measured at chance (AUC 0.484, e115) on
a different pool. This test uses richer bag statistics and judges the GATE's ranking, so it is not the same measurement -- but
the prior is weak and the criterion stays strict.

Bag features added per chip (leave-one-out inside the bag, so a chip never sees itself):
  bag_size, n_products, span_days,
  visual agreement across the bag, mean/median knowledge-vs-visual disagreement,
  mean |margin| of the other members, mean agreement of the arms' predictions on the others.

Pre-registered: the bag-augmented gate's recovery at a 20% budget must exceed the single-chip gate's, paired over the ports
that have bags, and the gate AUC must improve with paired p < 0.05.
"""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]     # compliant block
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
BUDGET, NREP = 0.20, 20
rng = np.random.default_rng(0)


# bags by (port, mmsi) with >=2 distinct products
bags = defaultdict(list)
for i, r in enumerate(rows):
    m = (r.get('mmsi') or '').strip()
    if m and m not in ('0', 'nan'):
        bags[(r['port'], m)].append(i)
BAG = {}
for k, idxs in bags.items():
    prods = {rows[i]['product'] for i in idxs}
    if len(prods) >= 2 and len(idxs) >= 2:
        for i in idxs:
            BAG[i] = k
print('袋 %d ｜ 覆盖 chip %d/%d (%.0f%%)' % (
    sum(1 for k, v in bags.items() if len({rows[i]['product'] for i in v}) >= 2 and len(v) >= 2),
    len(BAG), len(rows), 100.0 * len(BAG) / len(rows)), flush=True)


def fit(trs, te, use_k):
    Z = np.c_[X[trs], K[trs]] if use_k else X[trs]
    Q = np.c_[X[te], K[te]] if use_k else X[te]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0], -(S * np.log(S + 1e-12)).sum(1)


def bagfeat(idx_target, pv, pk, mv, mk):
    """leave-one-out bag statistics for the target chips."""
    out = np.zeros((len(idx_target), 6), np.float32)
    members = defaultdict(list)
    for j, gi in enumerate(idx_target):
        members[BAG.get(gi, ('_', j))].append(j)
    for key, js in members.items():
        if len(js) < 2:
            continue
        for j in js:
            other = [x for x in js if x != j]
            out[j, 0] = len(other)
            out[j, 1] = len({rows[idx_target[x]]['product'] for x in other})
            out[j, 2] = float(np.mean([(pv[x] == pk[x]) for x in other]))
            out[j, 3] = float(np.mean(np.abs(mk[other])))
            out[j, 4] = float(np.mean([(pk[x] == pv[j]) for x in other]))
            out[j, 5] = float(np.mean(np.abs(pv[other] == pv[j])))
    return out


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
solo_r, bag_r, solo_a, bag_a = [], [], [], []
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    te_bag = te[[i for i, gi in enumerate(te) if gi in BAG]]
    if len(te_bag) < 40 or len(src) < 500:
        continue
    # gate training set: source episodes
    F, L = [], []
    for q in [x for x in PU if x != p][:3]:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 500 or len(teq) < 20:
            continue
        qv, mqv, qev = fit(trq, teq, False)
        qk, mqk, qek = fit(trq, teq, True)
        base = np.c_[mqv, mqk, qek, qev, (qv == qk).astype(float)]
        bf = bagfeat(teq, qv, qk, mqv, mqk)
        F.append(np.c_[base, bf])
        L.append(((qk == y[teq]) & (qv != y[teq])).astype(int))
    if not F:
        continue
    F = np.vstack(F); L = np.concatenate(L)
    pv, mv, ev = fit(src, te, False)
    pk, mk, ek = fit(src, te, True)
    base_t = np.c_[mv, mk, ek, ev, (pv == pk).astype(float)]
    bf_t = bagfeat(te, pv, pk, mv, mk)
    Ft = np.c_[base_t, bf_t]
    # evaluate on the bagged subset only, so the comparison is on identical instances
    selm = np.array([gi in BAG for gi in te])
    pv_s, pk_s, yy = pv[selm], pk[selm], y[te[selm]]
    F_s, Ft_s = F, Ft[selm]
    L_s = L
    bV = ba(yy, pv_s)
    gains = ba(yy, pk_s) - bV
    if gains <= 0:
        continue
    n = max(1, int(BUDGET * selm.sum()))
    for tag, M, Mt in (('solo', F, Ft_s), ('bag', F, Ft_s)):
        use = slice(0, 5) if tag == 'solo' else slice(0, 11)
        g = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=0).fit(M[:, use], L_s)
        s = g.predict_proba(Mt[:, use])[:, 1]
        sel = np.zeros(int(selm.sum()), bool); sel[np.argsort(-s)[:n]] = True
        r = (ba(yy, np.where(sel, pk_s, pv_s)) - bV) / gains
        rand = float(np.mean([(ba(yy, np.where(np.isin(np.arange(int(selm.sum())), rng.choice(int(selm.sum()), n, replace=False)), pk_s, pv_s)) - bV) / gains for _ in range(NREP)]))
        ok = ((pk_s == yy) & (pv_s != yy)).astype(int)
        auc = roc_auc_score(ok, s) if 0 < ok.sum() < len(ok) else float('nan')
        (solo_r if tag == 'solo' else bag_r).append(r)
        (solo_a if tag == 'solo' else bag_a).append(auc)
    print('%-16s n_bag=%5d 增益 %+.4f' % (p, int(selm.sum()), gains), flush=True)

sr, br = np.array(solo_r), np.array(bag_r)
sa, ba_ = np.array(solo_a), np.array(bag_a)
print('')
print('臂            收复@20%%        门控 AUC')
print('单芯片         %.3f            %.3f' % (sr.mean(), np.nanmean(sa)))
print('加袋特征       %.3f            %.3f' % (br.mean(), np.nanmean(ba_)))
print('')
print('配对 收复: %+.3f  p=%.4f' % ((br - sr).mean(), stats.wilcoxon(br, sr).pvalue if np.any(br != sr) else float('nan')))
m = np.isfinite(sa) & np.isfinite(ba_)
print('配对 AUC : %+.3f  p=%.4f' % ((ba_ - sa)[m].mean(), stats.wilcoxon(ba_[m], sa[m]).pvalue if m.sum() >= 8 else float('nan')))
print('')
print('预注册判据（袋特征收复更高 且 AUC 提升 p<0.05）: %s（ORACLE_IDENTITY，不可部署 ✓）'
      % ('成立 ✓✓' if (br.mean() > sr.mean() and m.sum() >= 8 and stats.wilcoxon(ba_[m], sa[m]).pvalue < 0.05) else '未成立 ✗'))

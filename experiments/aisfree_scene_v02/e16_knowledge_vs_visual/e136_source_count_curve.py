"""e136: the source-port-count curve -- how much does the module gain per extra source port?

The pool was the single largest lever measured (8 -> 24 ports gave +14.0 pp on the c64 pool, E18h), and the paired tests kept
running into power limits at 24 ports. This quantifies the curve on the old pool with the COMPLIANT knowledge block (0:71), so
the relationship can be extrapolated instead of guessed. Two things are measured against the source count n_src:
  (a) the expert's gain: BA(V+K) - BA(V), and
  (b) the module's recovery at a 20% budget: (rank - random) / full-expert-gain.
Pre-registered: both increase monotonically in n_src. If they saturate before 23, more ports will not help further.
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]      # compliant
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
COUNTS = (2, 4, 8, 16, 23)
NREP, BUDGET, NINNER = 20, 0.20, 2
rng = np.random.default_rng(0)
PU = sorted(set(ports.tolist()))


def fit(trs, te, use_k):
    Z = np.c_[X[trs], K[trs]] if use_k else X[trs]
    Q = np.c_[X[te], K[te]] if use_k else X[te]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1); t2 = np.sort(L, 1)[:, -2:]
    return L.argmax(1), t2[:, 1] - t2[:, 0], -(S * np.log(S + 1e-12)).sum(1)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


print('%-8s %10s %14s %14s %10s' % ('n_src', 'ΔBA(pp)', 'rank@20%(pp)', 'random@20%(pp)', '收复'))
for ns in COUNTS:
    gains, ranks, rands = [], [], []
    for p in PU:
        oth = [x for x in PU if x != p]
        src = np.where(ports != p)[0]
        te = np.where(ports == p)[0]
        if len(te) < 20:
            continue
        if ns < len(oth):
            pick = oth[:ns] if ns <= 4 else list(rng.choice(oth, ns, replace=False))
        else:
            pick = oth
        tr = np.where(np.isin(ports, pick))[0]
        if len(tr) < 300:
            continue
        pv, mv, ev = fit(tr, te, False)
        pk, mk, ek = fit(tr, te, True)
        bV, bK = ba(y[te], pv), ba(y[te], pk)
        gains.append(bK - bV)
        # gate trained on source episodes
        F, L = [], []
        for q in pick[:NINNER]:
            trq = np.where(np.isin(ports, [x for x in pick if x != q]))[0]
            teq = np.where(ports == q)[0]
            if len(trq) < 300 or len(teq) < 20:
                continue
            qv, mqv, qev = fit(trq, teq, False)
            qk, mqk, qek = fit(trq, teq, True)
            F.append(np.c_[mqv, mqk, qek, qev, (qv == qk).astype(float)])
            L.append(((qk == y[teq]) & (qv != y[teq])).astype(int))
        if not F:
            continue
        F = np.vstack(F); L = np.concatenate(L)
        g = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=0).fit(F, L)
        s = g.predict_proba(np.c_[mv, mk, ek, ev, (pv == pk).astype(float)])[:, 1]
        n = max(1, int(BUDGET * len(te)))
        sel = np.zeros(len(te), bool); sel[np.argsort(-s)[:n]] = True
        # absolute pp, never a ratio: a near-zero per-port gain turns a ratio into noise
        ranks.append(100 * (ba(y[te], np.where(sel, pk, pv)) - bV))
        rands.append(100 * float(np.mean([(ba(y[te], np.where(np.isin(np.arange(len(te)), rng.choice(len(te), n, replace=False)), pk, pv)) - bV) for _ in range(NREP)])))
    print('%-8d %+10.2f %12.3f %12.3f %9.1f%%'
          % (ns, 100 * float(np.mean(gains)), float(np.mean(ranks)), float(np.mean(rands)),
             100 * float(np.mean(ranks))), flush=True)

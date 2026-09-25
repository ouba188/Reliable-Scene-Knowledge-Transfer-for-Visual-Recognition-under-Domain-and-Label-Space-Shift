"""e175 (takeover): two mechanisms compete for the same budget -- which one should spend it?

Both lines ended with a rank-plus-budget mechanism:
  A) the granularity line: rank by the FINE classifier's margin, decide at fine granularity for the top-k%
  B) the knowledge line:  rank by the consultation gate, consult the knowledge EXPERT for the top-k%
They act on different objects but they spend the same deployment budget, so the honest question is head-to-head: on the SAME
10/20% of instances, whose selected subset scores higher?

Reported per budget: BA of the selected subset under (a) fine-margin ranking with the plain classifier, (b) gate ranking with
the expert consulted, (c) a matched-size random subset as the floor. Paired over the 24 ports.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
ART = BASE / 'artifacts/eight_class_adaptive_20260916'
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]      # compliant
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
KS = (0.10, 0.20)
NREP, NINNER = 30, 2
rng = np.random.default_rng(0)


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


PU = sorted(set(ports.tolist()))
out = {k: {'fine': [], 'expert': [], 'rand': []} for k in KS}
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    pv, mv, ev = fit(src, te, False)
    pk, mk, ek = fit(src, te, True)
    # gate trained on inner source episodes, as the module does
    F, L = [], []
    for q in [x for x in PU if x != p][:NINNER]:
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
    s_gate = g.predict_proba(np.c_[mv, mk, ek, ev, (pv == pk).astype(float)])[:, 1]
    yy = y[te]
    for k in KS:
        n = max(1, int(k * len(te)))
        sel_fine = np.argsort(-mv)[:n]          # (a) the granularity line's own selector: the fine margin of the plain arm
        sel_exp = np.argsort(-s_gate)[:n]       # (b) the knowledge line's selector: the consultation gate
        out[k]['fine'].append(ba(yy[sel_fine], pv[sel_fine]))
        out[k]['expert'].append(ba(yy[sel_exp], pk[sel_exp]))
        out[k]['rand'].append(float(np.mean([(lambda idx: ba(yy[idx], pv[idx]))(rng.permutation(len(te))[:n])
                                             for _ in range(NREP)])))
    print('%-16s n=%5d 全量 V %.4f 专家 %.4f' % (p, len(te), ba(yy, pv), ba(yy, pk)), flush=True)

print('')
print('%-8s %12s %12s %12s %10s %10s' % ('budget', 'fine-margin', 'gate+expert', 'random', 'p(fine-exp)', 'p(exp-rand)'))
for k in KS:
    f, e, r = np.array(out[k]['fine']), np.array(out[k]['expert']), np.array(out[k]['rand'])
    p1 = stats.wilcoxon(f, e).pvalue if np.any(f != e) else float('nan')
    p2 = stats.wilcoxon(e, r).pvalue if np.any(e != r) else float('nan')
    print('%-8s %12.4f %12.4f %12.4f %10.4f %10.4f' % ('%.0f%%' % (100 * k), f.mean(), e.mean(), r.mean(), p1, p2))
print('')
for k in KS:
    f, e = np.array(out[k]['fine']), np.array(out[k]['expert'])
    win = 'expert ✓' if e.mean() > f.mean() else 'fine-margin ✓'
    print('预算 %2.0f%%：谁更值 ⇒ %s（%.4f vs %.4f，Δ %+.4f）' % (100 * k, win, e.mean(), f.mean(), (e - f).mean()))

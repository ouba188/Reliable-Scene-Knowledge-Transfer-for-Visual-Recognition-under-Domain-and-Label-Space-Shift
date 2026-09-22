"""e33: statistics + error decomposition for the key arms.

1) permutation test: 1000 shuffles of dK -> null distribution of
   (selection accuracy, corr(g, gain), gate BA) => p-values for E16e's criterion
2) Rescue / Harm / NetGain (per the literature note) for V vs V+K, and for the gate
   Rescue = P(K correct, V wrong); Harm = P(K wrong, V correct); NetGain = Rescue - Harm
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
name_of = {int(r['class_id']): r['class_name'] for r in rows}
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))
NP = 1000


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def preds(tr, te):
    A, B = feats(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    return cv.predict(X[te]), ck.predict(B)


def prior(pr):
    h = np.bincount(pr, minlength=C).astype(float)
    return h / max(1, h.sum())


recs = []
resc = harm = tot = 0
percls = {c: [0, 0, 0] for c in range(C)}       # [V correct, K correct, n]
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv, pk = preds(tr, te)
    pv_tr, _ = preds(tr, tr)
    bv, bk = ba(y[te], pv), ba(y[te], pk)
    okv, okk = (pv == y[te]), (pk == y[te])
    resc += int((okk & ~okv).sum()); harm += int((okv & ~okk).sum()); tot += len(te)
    for c in range(C):
        m = y[te] == c
        if m.any():
            percls[c][0] += int(okv[m].sum()); percls[c][1] += int(okk[m].sum()); percls[c][2] += int(m.sum())
    dK = np.zeros(C); nq = 0
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b = preds(itr, ite)
        for c in range(C):
            m = y[ite] == c
            if m.any():
                dK[c] += float((b[m] == c).mean()) - float((a[m] == c).mean())
        nq += 1
    dK /= max(1, nq)
    recs.append(dict(port=p, bv=bv, bk=bk, s=prior(pv) - prior(pv_tr), dK=dK))

g = np.array([r['s'] @ r['dK'] for r in recs])
gain = np.array([r['bk'] - r['bv'] for r in recs])
oracle = gain > 0
acc0 = float(np.mean((g > 0) == oracle))
cor0 = float(np.corrcoef(g, gain)[0, 1])
gate0 = float(np.mean([r['bk'] if g[i] > 0 else r['bv'] for i, r in enumerate(recs)]))

rng = np.random.default_rng(0)
null_acc, null_cor, null_gate = [], [], []
for _ in range(NP):
    gs = np.array([r['s'] @ r['dK'][rng.permutation(C)] for r in recs])
    null_acc.append(float(np.mean((gs > 0) == oracle)))
    null_cor.append(float(np.corrcoef(gs, gain)[0, 1]))
    null_gate.append(float(np.mean([r['bk'] if gs[i] > 0 else r['bv'] for i, r in enumerate(recs)])))
na, nc, ng = np.array(null_acc), np.array(null_cor), np.array(null_gate)
pv_acc = float((na >= acc0).mean())
pv_cor = float((nc >= cor0).mean())
pv_gate = float((ng >= gate0).mean())

print('=== 置换检验 (%d 次打乱 dK 的类顺序) ===' % NP)
print('  选择正确率: 实际 %.3f (%d/%d)  对照 mean %.3f  P95 %.3f  → p = %.4f' % (
    acc0, int(acc0 * len(recs)), len(recs), na.mean(), np.percentile(na, 95), pv_acc))
print('  corr(g,gain): 实际 %+.3f  对照 mean %+.3f  P95 %+.3f  → p = %.4f' % (
    cor0, nc.mean(), np.percentile(nc, 95), pv_cor))
print('  gate BA:     实际 %.4f  对照 mean %.4f  P95 %.4f  → p = %.4f' % (
    gate0, ng.mean(), np.percentile(ng, 95), pv_gate))
print()
print('=== Rescue / Harm / NetGain (V vs V+K, 全部样本) ===')
print('  Rescue = P(K对, V错) = %d/%d = %.4f' % (resc, tot, resc / tot))
print('  Harm   = P(K错, V对) = %d/%d = %.4f' % (harm, tot, harm / tot))
print('  NetGain= Rescue − Harm = %+.4f  (命中率净变化 %+.4f)' % ((resc - harm) / tot, (resc - harm) / tot))
print()
print('%-26s %7s %7s %8s %6s' % ('class', 'V rec', 'K rec', 'Δ', 'n'))
for c in range(C):
    vc, kc, n = percls[c]
    if n:
        print('%-26s %7.3f %7.3f %+8.3f %6d' % (name_of[c], vc / n, kc / n, (kc - vc) / n, n))

"""e27: pin down the prior-shift alignment criterion (e26) -- multi-seed shuffle control + full table.

Everything computed once (outer heads + source-side dK), then:
  - 5 shuffle seeds for dK's class order (control distribution)
  - correlation and selection accuracy per variant
  - full per-port table
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
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return np.c_[X[tr], (A - mu) / sd], np.c_[X[te], (B - mu) / sd]


def heads(tr, te):
    A, B = feats(tr, te)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(X[tr], y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(A, y[tr])
    return cv.predict(X[te]), ck.predict(B)


def prior(pred):
    h = np.bincount(pred, minlength=C).astype(float)
    return h / max(1, h.sum())


recs = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    pv_te, pk_te = heads(tr, te)
    pv_tr, _ = heads(tr, tr)
    bv, bk = ba(y[te], pv_te), ba(y[te], pk_te)
    dK = np.zeros(C); nq = 0
    for q in sorted(set(ports[tr].tolist())):
        itr = tr[ports[tr] != q]; ite = tr[ports[tr] == q]
        if len(ite) < 5 or len(set(y[itr].tolist())) < C:
            continue
        a, b = heads(itr, ite)
        for c in range(C):
            m = y[ite] == c
            if m.any():
                dK[c] += float((b[m] == c).mean()) - float((a[m] == c).mean())
        nq += 1
    dK /= max(1, nq)
    shiftV = prior(pv_te) - prior(pv_tr)
    recs.append(dict(port=p, bv=bv, bk=bk, g=float(shiftV @ dK), gK=float(prior(pk_te) @ dK),
                     dK=dK, dKm=np.linalg.norm(dK), svm=np.linalg.norm(shiftV),
                     shiftV=shiftV, oracle=bk > bv))

g = np.array([r['g'] for r in recs]); gain = np.array([r['bk'] - r['bv'] for r in recs])
pick = np.array([r['g'] > 0 for r in recs])
orc = np.array([r['oracle'] for r in recs])
print('=== 主判据 alignV ===')
print('  方向判对 %d/%d ; corr(g, gain) = %+.3f' % (int((pick == orc).sum()), len(recs), float(np.corrcoef(g, gain)[0, 1])))
print('  选 K 的港数: %d ; 实际净为正的港数: %d' % (int(pick.sum()), int(orc.sum())))
m_keep = float(np.mean([r['bk'] if r['g'] > 0 else r['bv'] for r in recs]))
print('  gate BA = %.4f  (V %.4f / always-K %.4f / oracle %.4f)' % (
    m_keep, np.mean([r['bv'] for r in recs]), np.mean([r['bk'] for r in recs]),
    np.mean([r['bk'] if r['oracle'] else r['bv'] for r in recs])))

print()
print('=== 多种子打乱对照（打乱 dK 的类顺序）===')
for s in range(5):
    rng = np.random.default_rng(100 + s)
    accs, cors, bas = [], [], []
    for r in recs:
        dKs = r['dK'][rng.permutation(C)]
        gs = float(r['shiftV'] @ dKs)
        accs.append((gs > 0) == r['oracle'])
        cors.append(gs)
        bas.append(r['bk'] if gs > 0 else r['bv'])
    print('  seed %d: 判对 %2d/%d  corr %+.3f  gate BA %.4f' % (
        s, sum(accs), len(recs), float(np.corrcoef(np.array(cors), gain)[0, 1]), float(np.mean(bas))))

print()
print('%-18s %6s %6s %9s %6s %6s %9s' % ('port', 'V', 'K', 'g_align', 'pick', 'true', 'shiftNorm'))
for r in sorted(recs, key=lambda x: -x['g']):
    print('%-18s %6.3f %6.3f %+9.4f %6s %6s %9.3f' % (
        r['port'], r['bv'], r['bk'], r['g'], 'K' if r['g'] > 0 else 'V',
        'K' if r['oracle'] else 'V', r['svm']))

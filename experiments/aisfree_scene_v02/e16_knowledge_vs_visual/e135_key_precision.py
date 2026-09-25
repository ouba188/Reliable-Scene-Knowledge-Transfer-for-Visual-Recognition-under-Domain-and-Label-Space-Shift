"""e135: is the class-key loss (49-57%) due to a COARSE key, or is it fundamental? (H1 vs H2)

The open-set risk experiment left a reproducible cost: keying the per-class threshold on the PREDICTED class recovered only
~half of what a TRUE-class key recovers. Two explanations, and the difference matters:
  H1 -- the key is too coarse (a single predicted label per instance): finer LABEL-FREE keys should recover more.
  H2 -- the order->value gap is fundamental: every label-free key stays at ~50%.

Keys tested (all label-free except the oracle):
  global          one threshold, no key
  predclass       threshold of the argmax class           (the measured baseline)
  confbucket      threshold of the predicted class AND the confidence tercile
  targetcluster   threshold of the k-means cluster the instance falls in (k=8, fitted on the TARGET features alone)
  trueclass       oracle key (upper bound)
Acceptance score: max softmax (the best-scoring option measured in e121). Metric: AURC over the target port, unknown
instances counted as errors. Pre-registered: if any label-free key other than predclass recovers >= 60% of the
global->trueclass gap (i.e. loss <= 40%), H1 holds; if all stay at ~50%, H2 holds.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier

ART = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
kk = np.load(ART / 'knowledge' / 'relations.npz')
K = np.where(kk['support'], kk['knowledge'], 0.0).astype(np.float32)[:, 0:71]     # compliant P0 AIS-free block
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
NINNER, ALPHA = 2, 0.05
PU = sorted(set(ports.tolist()))


def fit(trs, te):
    Z = np.c_[X[trs], K[trs]]; Q = np.c_[X[te], K[te]]
    mu = Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z - mu) / sd, y[trs])
    L = m.decision_function((Q - mu) / sd)
    S = softmax(L, 1)
    return S, S.argmax(1), S.max(1)


def thr(s, err, w=None, alpha=ALPHA):
    o = np.argsort(-s); s, err = s[o], err[o]
    if w is None:
        cum = np.cumsum(err) / (np.arange(len(err)) + 1)
    else:
        w = w[o]
        cum = np.cumsum(w * err) / np.cumsum(w)
    ok = np.where(cum <= alpha)[0]
    return float(s[ok.max()]) + 1e-9 if len(ok) else np.inf


def aurc(score, ok):
    o = np.argsort(-score)
    return float((np.cumsum(1 - ok[o]) / (np.arange(len(ok)) + 1)).mean())


KEYS = ('global', 'predclass', 'confbucket', 'targetcluster', 'trueclass')
res = {k: [] for k in KEYS}
for p in PU:
    src = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(src) < 500 or len(te) < 50:
        continue
    cs, cpr, ccf, ct, ci = [], [], [], [], []
    for q in [x for x in PU if x != p][:NINNER]:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 500 or len(teq) < 50:
            continue
        S, pr, cf = fit(trq, teq)
        cs.append(cf); cpr.append(pr); ccf.append(cf); ct.append(y[teq]); ci.append(teq)
    if not cs:
        continue
    cs = np.concatenate(cs); cpr = np.concatenate(cpr); ccf = np.concatenate(ccf); ct = np.concatenate(ct); cidx = np.concatenate(ci)
    cerr = (cpr != ct).astype(float)
    S, pr, cf = fit(src, te)
    yy = y[te]; ok = pr == yy                      # only known-class instances exist in this pool
    t_glob = thr(cs, cerr)
    qc = np.quantile(ccf, [1 / 3, 2 / 3])
    cb = np.digitize(ccf, qc)
    t_cb = {}
    for j in range(C):
        for b in range(3):
            m = (cpr == j) & (cb == b)
            t_cb[(j, b)] = thr(cs[m], cerr[m]) if m.sum() >= 60 else t_glob
    t_pc = {j: (thr(cs[cpr == j], cerr[cpr == j]) if (cpr == j).sum() >= 60 else t_glob) for j in range(C)}
    t_tc = {j: (thr(cs[ct == j], cerr[ct == j]) if (ct == j).sum() >= 60 else t_glob) for j in range(C)}
    km = KMeans(n_clusters=8, n_init=4, random_state=0).fit(X[te])
    lab_t = km.predict(X[te])
    t_cl = {}
    for c in range(8):
        m = lab_t == c
        if m.sum() >= 60:
            # calibrate the cluster key on source instances assigned to the nearest target cluster centre
            lab_s = km.predict(X[cidx])
            msk = lab_s == c
            t_cl[c] = thr(cs[msk], (cpr[msk] != ct[msk]).astype(float)) if msk.sum() >= 60 else t_glob
        else:
            t_cl[c] = t_glob
    cb_t = np.digitize(cf, qc)
    score = {'global': cf,
             'predclass': np.array([cf[i] - t_pc[pr[i]] for i in range(len(te))]),
             'confbucket': np.array([cf[i] - t_cb[(pr[i], cb_t[i])] for i in range(len(te))]),
             'targetcluster': np.array([cf[i] - t_cl[lab_t[i]] for i in range(len(te))]),
             'trueclass': np.array([cf[i] - t_tc[yy[i]] for i in range(len(te))])}
    for k in KEYS:
        res[k].append(aurc(score[k], ok))
    print('%-16s n=%5d 准确率 %.3f' % (p, len(te), ok.mean()), flush=True)

g = np.array(res['global']); t = np.array(res['trueclass'])
gap = (g - t).mean()
print('')
print('%-14s %10s %12s %10s' % ('key', 'AURC', 'Δ vs global', '收复'))
out = {}
for k in KEYS:
    v = np.array(res[k]); rec = 100 * (g - v).mean() / gap if gap else float('nan')
    out[k] = rec
    print('%-14s %10.4f %+12.4f %9.1f%%' % (k, v.mean(), (v - g).mean(), rec))
print('')
lab_free = [k for k in ('predclass', 'confbucket', 'targetcluster') if out[k] >= 60]
print('预注册判据（某无标签键收复 >=60%%）: %s ⇒ %s'
      % ('成立 ✓' if lab_free else '未成立 ✗',
         ('H1：键太粗，可用 %s 改善 ✓✓' % ','.join(lab_free)) if lab_free else 'H2：序→值损失是根本限额 ✗'))

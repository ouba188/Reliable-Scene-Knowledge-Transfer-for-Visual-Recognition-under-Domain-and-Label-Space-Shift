"""e25: two faithful-ish adaptations of the literature mechanisms, as label-free port selectors.

A) SENTRY-style PERTURBATION CONSISTENCY (SENTRY uses random image transforms + committee agreement;
   here we downgrade to feature-space perturbation, cf. WACV25 'Feature Space Perturbation'):
   for each candidate, perturb the standardized target features R times, measure how often the
   argmax prediction stays the same. More stable candidate wins.

B) MixVal-style CLUSTER intra/inter mixing (NeurIPS23): cluster the target features, mixup
   INTRA-cluster and INTER-cluster pairs, score = mean max-prob(intra) - mean max-prob(inter).
   (paper uses clusters, not predicted classes -- my earlier e24 used predicted classes)

Arms: V | K | sentry_pick | mixval_cluster_pick | oracle | rand
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHA = 0.3
PLIST = sorted(set(ports.tolist()))
rng = np.random.default_rng(17)
R_PERT = 20
SIGMA = 0.08


def ba(yy, pr):
    rs = [float((pr[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def build(tr, te):
    """candidate feature matrices, standardized on TRAIN stats"""
    A1 = K[tr][:, LEGAL]; B1 = K[te][:, LEGAL]
    mu = A1.mean(0); sd = A1.std(0) + 1e-9
    return (X[tr], X[te]), (np.c_[X[tr], (A1 - mu) / sd], np.c_[X[te], (B1 - mu) / sd])


def consistency(clf, Fte):
    """SENTRY-downgraded: fraction of samples whose argmax survives feature perturbation"""
    base = clf.predict(Fte)
    # 尺度对齐：按各维标准差扰动
    sd = Fte.std(0) + 1e-9
    agree = []
    for _ in range(R_PERT):
        pr = clf.predict(Fte + rng.normal(0, SIGMA, Fte.shape) * sd)
        agree.append(float((pr == base).mean()))
    return float(np.mean(agree))


def mixval_cluster(clf, Fte, k_cls=8):
    """MixVal-style: cluster the target, mix intra/inter-cluster, confidence gap"""
    n = len(Fte)
    lab = KMeans(n_clusters=min(k_cls, max(2, n // 5)), n_init=4, random_state=0).fit_predict(Fte)
    idx = [np.where(lab == c)[0] for c in range(lab.max() + 1)]
    intra, inter = [], []
    for a in range(len(idx)):
        for b in range(a, len(idx)):
            if len(idx[a]) < 2 or len(idx[b]) < 2:
                continue
            for _ in range(10):
                i = rng.choice(idx[a]); j = rng.choice(idx[b])
                m = (0.5 * Fte[i] + 0.5 * Fte[j])[None, :]
                # 用分类器在混合点上的相对置信（决策分的 max-min）
                d = clf.decision_function(m)[0]
                s = float(d.max() - d.min())
                (intra if a == b else inter).append(s)
    if not intra or not inter:
        return float('nan')
    return float(np.mean(intra) - np.mean(inter))


arms = {a: [] for a in ['V', 'K', 'sentry_pick', 'mixc_pick', 'oracle', 'rand']}
tbl = []
for p in PLIST:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
        continue
    (Xtr, Xte), (Ktr, Kte) = build(tr, te)
    # 候选特征统一标准化，保证扰动尺度可比
    def s(trf, tef):
        mu = trf.mean(0); sd = trf.std(0) + 1e-9
        return (trf - mu) / sd, (tef - mu) / sd
    Xtr_, Xte_ = s(Xtr, Xte)
    Ktr_, Kte_ = s(Ktr, Kte)
    cv = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(Xtr_, y[tr])
    ck = RidgeClassifier(alpha=ALPHA, class_weight='balanced').fit(Ktr_, y[tr])
    bv = ba(y[te], cv.predict(Xte_)); bk = ba(y[te], ck.predict(Kte_))
    cons_v, cons_k = consistency(cv, Xte_), consistency(ck, Kte_)
    mix_v, mix_k = mixval_cluster(cv, Xte_), mixval_cluster(ck, Kte_)
    pick_s = bk if cons_k > cons_v else bv
    pick_m = bk if mix_k > mix_v else bv
    pick_o = bk if bk > bv else bv
    pick_r = bk if rng.random() < 0.5 else bv
    arms['V'].append(bv); arms['K'].append(bk)
    arms['sentry_pick'].append(pick_s); arms['mixc_pick'].append(pick_m)
    arms['oracle'].append(pick_o); arms['rand'].append(pick_r)
    tbl.append((p, bv, bk, cons_v, cons_k, mix_v, mix_k, cons_k > cons_v, mix_k > mix_v, bk > bv))

b = float(np.mean(arms['V']))
print('%-12s %8s %10s' % ('arm', 'mean BA', 'Δ vs V'))
print('-' * 32)
for a in ['V', 'K', 'sentry_pick', 'mixc_pick', 'oracle', 'rand']:
    m = float(np.mean(arms[a]))
    print('%-12s %8.4f %+10.4f' % (a, m, m - b))
print()
print('%-18s %6s %6s | %7s %7s | %7s %7s' % ('port', 'V', 'K', 'consV', 'consK', 'mixV', 'mixK'))
for t in tbl:
    print('%-18s %6.3f %6.3f | %7.3f %7.3f | %7.3f %7.3f' % t[:7])
okS = sum(1 for t in tbl if t[7] == t[9]); okM = sum(1 for t in tbl if t[8] == t[9])
print('\nSENTRY-扰动一致性 选择正确 %d/%d ; MixVal-聚类 选择正确 %d/%d' % (okS, len(tbl), okM, len(tbl)))

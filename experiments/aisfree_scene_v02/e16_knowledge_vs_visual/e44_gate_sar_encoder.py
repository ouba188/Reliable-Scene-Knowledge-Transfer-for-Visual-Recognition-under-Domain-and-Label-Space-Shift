"""e44: does the instance-level gate survive a different backbone? (SAR-pretrained S1 encoder)

Same protocol as e42/e43 but X = per-fold PCA-128 of the SAR-pretrained features (features_s1b,
correct dB/224 preprocessing). Arms: V | K_all | gate_cur, plus the sign-predictor AUC.
Randomised PCA + float32 to stay inside this box's memory.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
S1B = Path(r'E:/临时会话/visual_reliable_baseline/features_s1b/resnet50_s1b.float16.npy')
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0]
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PORT_U = sorted(set(ports.tolist()))
NSRC = 8
F = np.load(S1B).astype(np.float32)
print('S1b features', F.shape, flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def prep(tr, te):
    p = PCA(n_components=128, svd_solver='randomized', random_state=0).fit(F[tr])
    A, B = p.transform(F[tr]), p.transform(F[te])
    sd = A.std(0) + 1e-9
    A, B = A / sd, B / sd
    ka, kb = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = ka.mean(0); s = ka.std(0); s[s < 1e-9] = 1.0
    return A, B, (ka - mu) / s, (kb - mu) / s


def fit_arm(tr, te, alpha, cache):
    A, B, Ka, Kb = cache
    clfv = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(A, y[tr])
    pv = clfv.predict(B); sv = clfv.decision_function(B)
    km = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(Ka, y[tr])
    sk = km.decision_function(Kb)
    pk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(np.c_[A, Ka], y[tr]).predict(np.c_[B, Kb])
    return pv, pk, sv, sk, Kb, B


def f_cur(sv, sk, Ks, pv):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    mk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk, dis, oh, Ks]


def tune_alpha(src, te):
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in src[:3]:
            tr = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
            if len(tr) < 200 or len(set(y[tr].tolist())) < C:
                continue
            sc.append(ba(y[teq], fit_arm(tr, teq, a, prep(tr, teq))[1]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


print('%-20s %7s %7s %7s %7s %6s' % ('port', 'V', 'K_all', 'gate', 'AUC', 'tau'))
print('-' * 62)
res, aucs = [], []
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    a = tune_alpha(src, te)
    F1, G, CV, CK, YV = [], [], [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200 or len(set(y[trq].tolist())) < C:
            continue
        pv, pk, sv, sk, Ks, _ = fit_arm(trq, teq, a, prep(trq, teq))
        g = np.zeros(len(teq)); g[(pv != y[teq]) & (pk == y[teq])] = 1; g[(pv == y[teq]) & (pk != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        F1.append(f_cur(sv[m], sk[m], Ks[m], pv[m])); G.append(g[m])
        CV.append((pv == y[teq])[m]); CK.append((pk == y[teq])[m]); YV.append(y[teq][m])
    if not G:
        continue
    G = np.concatenate(G); CV = np.concatenate(CV); CK = np.concatenate(CK); YV = np.concatenate(YV)
    ytr = (G > 0).astype(int)
    if ytr.min() == ytr.max():
        continue
    gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(np.vstack(F1), ytr)
    ps = gbm.predict_proba(np.vstack(F1))[:, 1]
    def gacc(sel):
        ok = np.where(sel, CK, CV).astype(bool)
        per = [float(ok[YV == c].mean()) for c in range(C) if (YV == c).sum() > 0]
        return float(np.mean(per)) if per else -1
    t = max(TAUS, key=lambda tt: gacc(ps > tt))
    tr = np.where(ports != p)[0]
    cache = prep(tr, te)
    pv, pk, sv, sk, Ks, _ = fit_arm(tr, te, a, cache)
    b = f_cur(sv, sk, Ks, pv)
    pt = gbm.predict_proba(b)[:, 1]
    g = np.zeros(len(te)); g[(pv != y[te]) & (pk == y[te])] = 1; g[(pv == y[te]) & (pk != y[te])] = -1
    m = g != 0
    try:
        auc = roc_auc_score((g[m] > 0).astype(int), pt[m]) if m.sum() > 10 and len(set((g[m] > 0).tolist())) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    aucs.append(auc)
    res.append((p, ba(y[te], pv), ba(y[te], pk), ba(y[te], np.where(pt > t, pk, pv)), auc, t))
    print('%-20s %7.3f %7.3f %7.3f %7.3f %6.2f' % res[-1], flush=True)

V = np.array([r[1] for r in res]); KA = np.array([r[2] for r in res]); GT = np.array([r[3] for r in res])
dK = (KA - V) * 100; dG = (GT - V) * 100
print()
print('SAR 预训练编码器（S1b）上：')
print('  V        %.4f' % V.mean())
print('  K 全用    %.4f  Δ %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (KA.mean(), dK.mean(), (dK > 0).sum(), len(dK), dK.min()))
print('  K 闸门    %.4f  Δ %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (GT.mean(), dG.mean(), (dG > 0).sum(), len(dG), dG.min()))
from scipy import stats
print('  配对 闸门 − 全用: %+.2f pp  Wilcoxon p=%.4f  更好 %d / 更差 %d'
      % ((GT - KA).mean() * 100, stats.wilcoxon(GT, KA).pvalue, int((GT > KA).sum()), int((GT < KA).sum())))
av = np.array([a for a in aucs if np.isfinite(a)])
print('  G 符号预测 AUC: 平均 %.3f 中位 %.3f  >0.6 的港 %d/%d' % (av.mean(), np.median(av), (av > 0.6).sum(), len(av)))

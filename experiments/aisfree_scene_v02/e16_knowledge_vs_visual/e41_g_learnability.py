"""e41: the gate -- can the instance-level net gain G_i be predicted from label-free observables?

G_i = +1 rescue (V wrong, V+K right) | -1 harm (V right, V+K wrong) | 0 neutral.
Predictor inputs (all label-free): visual margin/entropy, predicted class, knowledge-only margin,
knowledge-visual disagreement, standardised knowledge vector r.
Train on the SOURCE ports' G samples (nested LOO), test on the held-out TARGET port -> AUC.

AUC ~ 0.5  => rho is unlearnable; retrieval/non-parametric Psi is equally hopeless.
AUC > 0.6  => structure exists; implement the gate / retrieval-based Psi.

Baselines: the visual margin alone (the "uncertainty" heuristic), the disagreement alone, chance.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0, 10.0]
PORT_U = sorted(set(ports.tolist()))
NSRC = 8                     # source ports used to train the G predictor


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fit_arm(tr, te, alpha):
    """returns (pred_v, pred_k, sv, sk, Ks_te)"""
    A, B = kstd(tr, te)
    clf_v = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(X[tr], y[tr])
    clf_k = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(A, y[tr])
    fk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(np.c_[X[tr], A], y[tr])
    sv = clf_v.decision_function(X[te]); sk = clf_k.decision_function(B)
    return clf_v.predict(X[te]), fk.predict(np.c_[X[te], B]), sv, sk, B


def gfeats(sv, sk, Ks, pred_v):
    p = softmax(sv, axis=1)
    srt = np.sort(sv, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    ent = -(p * np.log(p + 1e-12)).sum(1)
    oh = np.zeros((len(pred_v), C - 1)); oh[np.arange(len(pred_v)), np.minimum(pred_v, C - 2)] = 1
    margk = np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2]
    dis = (sk.argmax(1) != pred_v).astype(float)[:, None]
    return np.c_[marg[:, None], ent[:, None], margk[:, None], dis, oh, Ks]


def tune_alpha(src):
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in src[:3]:
            tr = np.where(np.isin(ports, src) & (ports != q))[0]; te = np.where(ports == q)[0]
            if len(tr) < 200 or len(set(y[tr].tolist())) < C:
                continue
            sc.append(ba(y[te], fit_arm(tr, te, a)[1]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


print('%-20s %6s %6s %7s %7s %7s %7s' % ('target port', 'n+', 'n-', 'AUC_gbm', 'AUC_lr', 'AUC_marg', 'AUC_dis'))
print('-' * 74)
res = []
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    a = tune_alpha(src)
    Gtr, Ftr = [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200 or len(set(y[trq].tolist())) < C:
            continue
        pv, pk, sv, sk, Ks = fit_arm(trq, teq, a)
        g = np.zeros(len(teq)); g[(pv != y[teq]) & (pk == y[teq])] = 1; g[(pv == y[teq]) & (pk != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        Gtr.append(g[m]); Ftr.append(gfeats(sv[m], sk[m], Ks[m], pv[m]))
    if not Gtr:
        continue
    Gtr = np.concatenate(Gtr); Ftr = np.vstack(Ftr)
    tr = np.where(ports != p)[0]
    pv, pk, sv, sk, Ks = fit_arm(tr, te, a)
    g = np.zeros(len(te)); g[(pv != y[te]) & (pk == y[te])] = 1; g[(pv == y[te]) & (pk != y[te])] = -1
    m = g != 0
    Ft = gfeats(sv[m], sk[m], Ks[m], pv[m]); gt = (g[m] > 0).astype(int)
    if gt.min() == gt.max() or (Gtr > 0).all() or (Gtr < 0).all():
        continue
    ytr = (Gtr > 0).astype(int)
    aucs = {}
    try:
        aucs['gbm'] = roc_auc_score(gt, HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                                      random_state=0).fit(Ftr, ytr).predict_proba(Ft)[:, 1])
    except Exception:
        aucs['gbm'] = float('nan')
    try:
        aucs['lr'] = roc_auc_score(gt, RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ftr, ytr)
                                   .decision_function(Ft))
    except Exception:
        aucs['lr'] = float('nan')
    try:
        aucs['marg'] = roc_auc_score(gt, -Ft[:, 0])      # 视觉越不自信 => 越可能救回
        aucs['dis'] = roc_auc_score(gt, Ft[:, 3])        # 知识与视觉分歧 => 越可能救回
    except Exception:
        aucs['marg'] = aucs['dis'] = float('nan')
    res.append((p, int(gt.sum()), int((1 - gt).sum()), aucs))
    print('%-20s %6d %6d %7.3f %7.3f %7.3f %7.3f' % (
        p, gt.sum(), (1 - gt).sum(), aucs['gbm'], aucs['lr'], aucs['marg'], aucs['dis']))

print()
for key, lab in [('gbm', 'GBM 非线性(z,r)'), ('lr', '线性(z,r)'), ('marg', '仅视觉不确定性'), ('dis', '仅知识-视觉分歧')]:
    v = np.array([r[3][key] for r in res])
    v = v[np.isfinite(v)]
    print('%-22s 平均 AUC %.3f  中位 %.3f  >0.6 的港 %d/%d  >0.55 的港 %d/%d'
          % (lab, v.mean(), np.median(v), (v > 0.6).sum(), len(v), (v > 0.55).sum(), len(v)))
npos = sum(r[1] for r in res); nneg = sum(r[2] for r in res)
print()
print('样本量: rescue %d / harm %d （共 %d 个 non-neutral 实例，%d 个港）' % (npos, nneg, npos + nneg, len(res)))

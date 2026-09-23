"""e56: closed-set structural value of the direction-aware correction operator.

Arms (all per-port LOO on the existing 38,091 / 128-d visual features / 71-d legal knowledge):
  V            visual ridge
  gate         current best (within-port percentile knowledge + instance gate + class-aware delta=0.2)
  joint_gbm    equal-capacity joint classifier on [z, k]  (the anti-'more capacity' control)
  U_argmax     learn pairwise gains U_{a->b}(s), then pick argmax_b of the gain from the visual's a
  U_op         full mass-conserving operator: r_ab = kappa*[U_ab]_+ ; F_ab = p_a r_ab/(1+sum_c r_ac)
  U_op_nok     same operator but the knowledge is zeroed (proves the gain comes from knowledge)

Pre-registered failure criteria (from the reviewed proposal):
  - only beating the old gate (not joint_gbm)      -> capacity effect only
  - U_op_nok improving just as much                -> no knowledge contribution
  - worst ports not replicating                    -> no transfer
  - winning only by rejecting/abstaining more       -> not applicable in closed set
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DELTA = 0.2
PORT_U = sorted(set(ports.tolist())); NSRC = 8
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
ARMS = ['V', 'gate', 'joint_gbm', 'U_argmax', 'U_op', 'U_op_nok']


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def fit_core(tr, te, a):
    A, B = kstd(tr, te)
    fv = RidgeClassifier(alpha=a, class_weight='balanced').fit(X[tr], y[tr])
    dv = fv.decision_function(X[te])
    pk = RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    sk = RidgeClassifier(alpha=a, class_weight='balanced').fit(A, y[tr]).decision_function(B)
    return fv, dv, pk, sk, A, B


def f_cur(sv, sk, Ks, pv):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    mk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk, dis, oh, Ks]


def rarity(pred):
    cnt = np.bincount(pred, minlength=C).astype(float); cnt[cnt == 0] = 1.0
    w = 1.0 / cnt
    return w / w.mean()


def gain_feats(X_, Ks_, a_pred):
    oh = np.zeros((len(a_pred), C)); oh[np.arange(len(a_pred)), a_pred] = 1
    return np.c_[X_, Ks_, oh]


def operator(U, p, kappa=1.0, topk=3):
    """U: (n, C) gain of moving from the current argmax a to each b. Mass-conserving update."""
    a = p.argmax(1)
    r = np.zeros_like(U)
    for i in range(len(p)):
        u = np.maximum(U[i], 0)
        u[a[i]] = 0
        if topk and topk < C:                       # 只保留最强的若干方向（避免噪声铺开）
            thr = np.sort(u)[-topk]
            u = np.where(u >= thr, u, 0.0)
        r[i] = kappa * u
    F = p[np.arange(len(p)), a][:, None] * r / (1.0 + r.sum(1, keepdims=True))
    out = p.copy()
    for i in range(len(p)):                          # 守恒修正
        out[i] += F[i]
        out[i, a[i]] -= F[i].sum()
    return np.clip(out, 0, None)


print('%-18s %6s %6s %6s %6s %6s %6s' % tuple(['port'] + ARMS))
print('-' * 74)
out = {a_: [] for a_ in ARMS}; pp = {a_: {} for a_ in ARMS}
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]; tr = np.where(ports != p)[0]
    if len(te) < 20:
        continue
    a = 1.0; best = -1
    for al in ALPHAS:
        sc = []
        for q in src[:3]:
            itr = np.where(np.isin(ports, src) & (ports != q))[0]; ite = np.where(ports == q)[0]
            if len(itr) < 200:
                continue
            sc.append(ba(y[ite], fit_core(itr, ite, al)[2]))
        if sc and float(np.mean(sc)) > best:
            best, a = float(np.mean(sc)), al
    # ---- 源端（嵌套 LOO）构造 U 的训练样本 ----
    FU, TU, G, rec = [], [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        fv, dv, pk, sk, Aq, Bq = fit_core(trq, teq, a)
        pvq = fv.predict(X[teq])
        prob = softmax(dv)
        U = np.zeros((len(teq), C))                 # 目标：L(a,y)-L(b,y)，0-1 损失下为 0/±1
        for i in range(len(teq)):                    # U[i,b] = L(a,y)-L(b,y)，0-1 损失
            a_i = pvq[i]
            U[i] = (np.arange(C) == y[teq][i]).astype(float) - (np.arange(C) == a_i).astype(float)
            U[i, a_i] = 0.0
        FU.append(gain_feats(X[teq], Bq, pvq)); TU.append(U)
        rec.append((fv, dv, pk, sk, Bq, pvq, teq)); G.append(np.zeros(len(teq)))
    FU = np.vstack(FU); TU = np.vstack(TU)
    est = Ridge(alpha=1.0).fit(FU, TU)               # 共享的小模型（线性，参数量 ≈ 联合分类器）
    # ---- 目标港 ----
    fv, dv, pk, sk, A, B = fit_core(tr, te, a)
    pv = fv.predict(X[te]); prob = softmax(dv)
    Uhat = est.predict(gain_feats(X[te], B, pv))
    # gate (current best baseline)
    F1, GG = [], []
    for (fvq, dvq, pkq, skq, Bq, pvq, teq) in rec:
        g = np.zeros(len(teq)); g[(pvq != y[teq]) & (pkq == y[teq])] = 1; g[(pvq == y[teq]) & (pkq != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        F1.append(f_cur(dvq[m], skq[m], Bq[m], pvq[m])); GG.append(g[m])
    gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(
        np.vstack(F1), (np.concatenate(GG) > 0).astype(int))
    pt = gbm.predict_proba(f_cur(dv, sk, B, pv))[:, 1]
    rc = rarity(pv); r_k = np.array([rc[c] for c in pk])
    gated = np.where((pt > 0.5 + DELTA * r_k) | (pk == pv), pk, pv)
    joint = HistGradientBoostingClassifier(max_iter=300, max_depth=4, random_state=0).fit(
        np.c_[X[tr], kstd(tr, tr)[0]], y[tr]).predict(np.c_[X[te], B])
    est_nok = Ridge(alpha=1.0).fit(np.c_[FU[:, :128], np.zeros((len(FU), 71)), FU[:, 199:]], TU)
    Uhat_nok = est_nok.predict(np.c_[X[te], np.zeros((len(te), 71)), gain_feats(X[te], B, pv)[:, 128 + 71:]])
    preds = {'V': pv, 'gate': gated, 'joint_gbm': joint,
             'U_argmax': Uhat.argmax(1), 'U_op': operator(Uhat, prob).argmax(1),
             'U_op_nok': operator(Uhat_nok, prob).argmax(1)}
    for nm, pr in preds.items():
        out[nm].append(ba(y[te], pr)); pp[nm][p] = out[nm][-1]
    print('%-18s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' % (p, *[out[nm][-1] for nm in ARMS]), flush=True)

print()
base = np.array(out['V'])
for nm in ARMS[1:]:
    v = np.array(out[nm]); d = (v - base) * 100
    print('%-10s mean %.4f  Δ %+6.2f  正港 %2d/%d  负港 %2d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), int((d < 0).sum()), d.min()))
gt = np.array(out['gate'])
print()
for nm in ['joint_gbm', 'U_argmax', 'U_op', 'U_op_nok']:
    v = np.array(out[nm]); dd = (v - gt) * 100
    try:
        pw = stats.wilcoxon(v, gt).pvalue
    except Exception:
        pw = float('nan')
    print('配对 %-10s − gate: %+6.2f pp  Wilcoxon p=%.4f' % (nm, dd.mean(), pw))

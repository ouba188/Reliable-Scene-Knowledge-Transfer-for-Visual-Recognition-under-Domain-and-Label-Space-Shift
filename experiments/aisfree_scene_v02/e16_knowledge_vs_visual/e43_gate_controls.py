"""e43: gate controls + improvements, all arms sharing the same fits.

Arms (all per-port LOO, same protocol as e16d/e41/e42):
  V              visual only
  K_all          always-on fusion (the thing to beat)
  gate_rand      random gate with the SAME enable rate as gate_cur   <- dilution control
  gate_marg      gate on the visual margin alone                     <- heuristic control
  gate_cur       the e42 gate (margin/entropy/class/knowledge-only/disagreement/r)
  gate_z         same + the full visual z (PCA-128)
  gate_z_tail    gate_z but tau chosen by the SOURCE worst-port BA   <- tail-optimising
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0]
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PORT_U = sorted(set(ports.tolist()))
NSRC = 8
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = K[tr][:, LEGAL], K[te][:, LEGAL]
    mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def fit_arm(tr, te, alpha):
    A, B = kstd(tr, te)
    clfv = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(X[tr], y[tr])
    pv = clfv.predict(X[te]); sv = clfv.decision_function(X[te])
    km = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(A, y[tr])
    sk = km.decision_function(B)
    pk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    return pv, pk, sv, sk, B, X[te]


def f_cur(sv, sk, Ks, pv):
    p = softmax(sv, axis=1); srt = np.sort(sv, axis=1)
    marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-12)).sum(1))[:, None]
    mk = (np.sort(sk, axis=1)[:, -1] - np.sort(sk, axis=1)[:, -2])[:, None]
    dis = (sk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk, dis, oh, Ks]


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


def gated_acc(cor_v, cor_k, sel, yv):
    ok = np.where(sel, cor_k, cor_v).astype(bool)
    per = [float(ok[yv == c].mean()) for c in range(C) if (yv == c).sum() > 0]
    return float(np.mean(per)) if per else float('nan')


ARMS = ['gate_rand', 'gate_marg', 'gate_cur', 'gate_z', 'gate_z_tail']
out = {a: [] for a in ['V', 'K_all'] + ARMS}
print('%-20s %7s %7s %7s %7s %7s %7s %7s' % ('port', 'V', 'K_all', 'g_rand', 'g_marg', 'g_cur', 'g_z', 'g_z_tail'))
print('-' * 76)
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    a = tune_alpha(src)
    F1, F2, G, CV, CK, YV, PV_ = [], [], [], [], [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200 or len(set(y[trq].tolist())) < C:
            continue
        pv, pk, sv, sk, Ks, Xw = fit_arm(trq, teq, a)
        g = np.zeros(len(teq)); g[(pv != y[teq]) & (pk == y[teq])] = 1; g[(pv == y[teq]) & (pk != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        base = f_cur(sv[m], sk[m], Ks[m], pv[m])
        F1.append(base); F2.append(np.c_[base, Xw[m]])
        G.append(g[m]); CV.append((pv == y[teq])[m]); CK.append((pk == y[teq])[m]); YV.append(y[teq][m])
        PV_.append(pv[m])
    if not G:
        continue
    G = np.concatenate(G); CV = np.concatenate(CV); CK = np.concatenate(CK); YV = np.concatenate(YV)
    ytr = (G > 0).astype(int)
    m1 = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(np.vstack(F1), ytr)
    m2 = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(np.vstack(F2), ytr)
    ps1 = m1.predict_proba(np.vstack(F1))[:, 1]; ps2 = m2.predict_proba(np.vstack(F2))[:, 1]
    marg_s = np.vstack(F1)[:, 0]
    t_mean = max(TAUS, key=lambda t: gated_acc(CV, CK, ps1 > t, YV))
    def worst_cls(sel):
        ok = np.where(sel, CK, CV).astype(bool)
        per = [float(ok[YV == c].mean()) for c in range(C) if (YV == c).sum() >= 5]
        return min(per) if per else -1.0
    t_tail = max(TAUS, key=worst_cls)        # 尾部目标：最大化源端最差类 BA
    tr = np.where(ports != p)[0]
    pv, pk, sv, sk, Ks, Xw = fit_arm(tr, te, a)
    b = f_cur(sv, sk, Ks, pv)
    pt1 = m1.predict_proba(b)[:, 1]; pt2 = m2.predict_proba(np.c_[b, Xw])[:, 1]
    mt = b[:, 0]
    sel_cur = pt1 > t_mean
    sel_z = pt2 > t_mean
    sel_zt = pt2 > t_tail
    rate = sel_cur.mean()
    sel_rnd = rng.random(len(te)) < rate
    med = np.median(mt)
    sel_marg = mt <= med                       # 用视觉不确定性：margin 低于中位数则用知识
    vals = {
        'V': ba(y[te], pv), 'K_all': ba(y[te], pk),
        'gate_rand': ba(y[te], np.where(sel_rnd, pk, pv)),
        'gate_marg': ba(y[te], np.where(sel_marg, pk, pv)),
        'gate_cur': ba(y[te], np.where(sel_cur, pk, pv)),
        'gate_z': ba(y[te], np.where(sel_z, pk, pv)),
        'gate_z_tail': ba(y[te], np.where(sel_zt, pk, pv)),
    }
    for kk, vv in vals.items():
        out[kk].append(vv)
    print('%-20s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f' % (
        p, vals['V'], vals['K_all'], vals['gate_rand'], vals['gate_marg'], vals['gate_cur'],
        vals['gate_z'], vals['gate_z_tail']))

print()
base = np.array(out['V'])
for a in ['K_all'] + ARMS:
    v = np.array(out[a]); d = (v - base) * 100
    print('%-12s mean BA %.4f  Δ %+6.2f  逐港正 %2d/%d  负港 %2d  worst %+6.2f'
          % (a, v.mean(), d.mean(), (d > 0).sum(), len(d), (d < 0).sum(), d.min()))
print()
from scipy import stats
ka = np.array(out['K_all'])
for a in ARMS:
    v = np.array(out[a]); dd = (v - ka) * 100
    try:
        p = stats.wilcoxon(v, ka).pvalue
    except Exception:
        p = float('nan')
    print('配对 %-12s − K_all: %+6.2f pp  Wilcoxon p=%.4f  更好 %d / 更差 %d'
          % (a, dd.mean(), p, (dd > 0).sum(), (dd < 0).sum()))

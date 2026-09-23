"""e50: three label-free rescue mechanisms for the remaining negative ports.

All three are selected ONLY on the source ports (never on the target -> no oracle).

(A) rate-matched gate: instead of an absolute tau, enable the top r*% of the target's chips by
    predicted rescue-probability, where r* is the enable rate that was optimal on the source ports.
(B) port safety valve: aggregate the instance-level predictions to the port. A target port is only
    allowed to use knowledge if its mean predicted rescue-probability is >= min over source ports
    whose gated arm actually beat V (a support-range rule, not a magic threshold).
(C) direction-consistency restriction: the knowledge may only flip a prediction to a class whose
    source-side per-class gain is positive on average (else keep the visual prediction).

Arms: V | K_pct | gate | gate+A | gate+B | gate+C | gate+ABC
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import softmax
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PORT_U = sorted(set(ports.tolist())); NSRC = 8
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
ARMS = ['V', 'K_pct', 'gate', 'gateA', 'gateB', 'gateC', 'gateABC']


def ba_mask(ok, yy):
    per = [float(ok[yy == c].mean()) for c in range(C) if (yy == c).sum() > 0]
    return float(np.mean(per)) if per else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def fit_arm(tr, te, alpha):
    A, B = kstd(tr, te)
    clfv = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(X[tr], y[tr])
    sk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(A, y[tr]).decision_function(B)
    pk = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    return clfv.predict(X[te]), pk, clfv.decision_function(X[te]), sk


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
            sc.append(ba_mask(fit_arm(tr, te, a)[1] == y[te], y[te]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


print('%-18s %6s %6s %6s %6s %6s %6s %6s' % tuple(['port'] + ARMS))
print('-' * 76)
out = {a: [] for a in ARMS}; per_port = {a: {} for a in ARMS}
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    a = tune_alpha(src)
    F1, G, rec = [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200 or len(set(y[trq].tolist())) < C:
            continue
        pv, pk, sv, sk = fit_arm(trq, teq, a)
        g = np.zeros(len(teq)); g[(pv != y[teq]) & (pk == y[teq])] = 1; g[(pv == y[teq]) & (pk != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        F1.append(f_cur(sv[m], sk[m], kstd(trq, teq)[1][m], pv[m])); G.append(g[m])
        rec.append((q, pv, pk, teq))
    if not G:
        continue
    ytr = (np.concatenate(G) > 0).astype(int)
    gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(np.vstack(F1), ytr)
    # ---- source-side statistics (all label-free at deploy time) ----
    ps_all, port_of, v_ok, k_ok, yv = [], [], [], [], []
    for (q, pv, pk, teq) in rec:
        _, _, svq, skq = fit_arm(np.where(np.isin(ports, src) & (ports != q))[0], teq, a)
        b = f_cur(svq, skq, kstd(np.where(np.isin(ports, src) & (ports != q))[0], teq)[1], pv)
        ps_all.append(gbm.predict_proba(b)[:, 1]); port_of += [q] * len(teq)
        v_ok.append(pv == y[teq]); k_ok.append(pk == y[teq]); yv.append(y[teq])
    ps_all = np.concatenate(ps_all); port_of = np.array(port_of)
    v_ok = np.concatenate(v_ok); k_ok = np.concatenate(k_ok); yv = np.concatenate(yv)
    rate, best = 0.5, -1
    for t in TAUS:
        s = ba_mask(np.where(ps_all > t, k_ok, v_ok), yv)
        if s > best:
            best, rate = s, float((ps_all > t).mean())
    pbar = {q: float(ps_all[port_of == q].mean()) for q in set(port_of)}
    dq = {q: ba_mask(np.where(ps_all[port_of == q] > 0.5, k_ok[port_of == q], v_ok[port_of == q]), yv[port_of == q])
          - ba_mask(v_ok[port_of == q], yv[port_of == q]) for q in set(port_of)}
    gain = [pbar[q] for q in pbar if dq.get(q, 0) > 0]
    pbar_min = min(gain) if gain else 0.0
    def gated_ok(q):
        mq = port_of == q
        return np.where(ps_all[mq] > 0.5, k_ok[mq], v_ok[mq])

    dyset = {}
    for c in range(C):
        vals = []
        for q in set(port_of):
            mq = (port_of == q) & (yv == c)
            if mq.sum() < 5:
                continue
            vals.append(float((np.where(ps_all[mq] > 0.5, k_ok[mq], v_ok[mq]) == yv[mq]).mean()
                              - (v_ok[mq] == yv[mq]).mean()))
        dyset[c] = float(np.mean(vals)) if vals else 0.0
    unsafe = {c for c, d in dyset.items() if d < 0} or {0}
    # ---- target ----
    tr = np.where(ports != p)[0]
    pv, pk, sv, sk = fit_arm(tr, te, a)
    b = f_cur(sv, sk, kstd(tr, te)[1], pv)
    pt = gbm.predict_proba(b)[:, 1]
    thr = np.quantile(pt, 1 - rate) if 0 < rate < 1 else 0.5
    selA = pt >= thr
    selQ = pt > 0.5
    pbar_t = float(pt.mean())
    keepB = pbar_t >= pbar_min
    # C: refuse flips towards classes whose source-side direction is negative
    pkC = np.where((pk != pv) & np.isin(pk, list(unsafe)), pv, pk)
    arms = {
        'V': pv, 'K_pct': pk,
        'gate': np.where(selQ, pk, pv),
        'gateA': np.where(selA, pk, pv),
        'gateB': np.where(selQ, pk, pv) if keepB else pv,
        'gateC': np.where(selQ, pkC, pv),
        'gateABC': (np.where(selA, pkC, pv) if keepB else pv),
    }
    for nm, pred in arms.items():
        v_ = ba_mask(pred == y[te], y[te]); out[nm].append(v_); per_port[nm][p] = v_
    print('%-18s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f  [rate %.2f pbar_t %.3f pbar* %.3f %s]' % (
        p, *[out[nm][-1] for nm in ARMS], rate, pbar_t, pbar_min, 'KEEP' if keepB else 'VALVE'), flush=True)

print()
base = np.array(out['V'])
for nm in ARMS[1:]:
    v = np.array(out[nm]); d = (v - base) * 100
    print('%-9s mean %.4f  Δ %+6.2f  正港 %2d/%d  负港 %2d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), int((d < 0).sum()), d.min()))
g = np.array(out['gate'])
for nm in ['gateA', 'gateB', 'gateC', 'gateABC']:
    v = np.array(out[nm]); dd = (v - g) * 100
    try:
        pv_ = stats.wilcoxon(v, g).pvalue
    except Exception:
        pv_ = float('nan')
    print('配对 %-8s − gate: %+6.2f pp  Wilcoxon p=%.4f' % (nm, dd.mean(), pv_))
print()
neg = [p for p in per_port['gateABC'] if per_port['gateABC'][p] < per_port['V'][p]]
print('gateABC 的负港:', ', '.join('%s(%+.2f)' % (p, (per_port['gateABC'][p] - per_port['V'][p]) * 100) for p in neg) or '无')

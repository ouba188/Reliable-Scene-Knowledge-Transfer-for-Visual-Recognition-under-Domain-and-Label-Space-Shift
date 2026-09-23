"""e53: full-LOO validation of the class-aware gate (the Singapore rescue, de-overfitted).

Same protocol as e50. delta is chosen ON THE SOURCE PORTS ONLY (never the target). Reported arms:
  V | K_pct | gate(0.5) | gate+classAware(delta grid, fixed) | gate+classAware(delta chosen on source)
If the fixed-delta arms help everywhere (or at least the source-selected one does), the rescue is a
mechanism; if only Singapore improves, it was overfitting.
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
PORT_U = sorted(set(ports.tolist())); NSRC = 8
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.5]
TAU = 0.5
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
ARMS = ['V', 'K_pct', 'gate', 'd10', 'd20', 'd30', 'srcSel']


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def fit(tr, te, a):
    A, B = kstd(tr, te)
    f = RidgeClassifier(alpha=a, class_weight='balanced').fit(X[tr], y[tr])
    km = RidgeClassifier(alpha=a, class_weight='balanced').fit(A, y[tr])
    pk = RidgeClassifier(alpha=a, class_weight='balanced').fit(np.c_[X[tr], A], y[tr]).predict(np.c_[X[te], B])
    return f.predict(X[te]), pk, f.decision_function(X[te]), km.decision_function(B), B


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


def tuned_alpha(src):
    best_a, best = 1.0, -1
    for al in ALPHAS:
        sc = []
        for q in src[:3]:
            tr = np.where(np.isin(ports, src) & (ports != q))[0]; te = np.where(ports == q)[0]
            if len(tr) < 200:
                continue
            sc.append(ba(y[te], fit(tr, te, al)[1]))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), al
    return best_a


def apply_rule(pv, pk, pt, delta):
    rc = rarity(pv)
    r_k = np.array([rc[c] for c in pk])
    sel = (pt > TAU + delta * r_k) | (pk == pv)
    return np.where(sel, pk, pv)


print('%-18s %6s %6s %6s %6s %6s %6s %6s' % tuple(['port'] + ARMS))
print('-' * 74)
out = {a: [] for a in ARMS}; pp = {a: {} for a in ARMS}
for p in PORT_U:
    src = [q for q in PORT_U if q != p]
    te = np.where(ports == p)[0]; tr = np.where(ports != p)[0]
    if len(te) < 20:
        continue
    a = tuned_alpha(src)
    F1, G, srcv, srck, srcpt, srcy = [], [], [], [], [], []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        pvq, pkq, svq, skq, Ksq = fit(trq, teq, a)
        g = np.zeros(len(teq)); g[(pvq != y[teq]) & (pkq == y[teq])] = 1; g[(pvq == y[teq]) & (pkq != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        F1.append(f_cur(svq[m], skq[m], Ksq[m], pvq[m])); G.append(g[m])
    if not G:
        continue
    gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0).fit(
        np.vstack(F1), (np.concatenate(G) > 0).astype(int))
    # --- source-side delta selection: fit ONCE per source port, then apply each delta to the cached
    #     predictions (the delta only changes how the rule is applied, not the model) ---
    cached = []
    for q in src[:NSRC]:
        trq = np.where(np.isin(ports, src) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        pvq, pkq, svq, skq, Ksq = fit(trq, teq, a)
        ptq = gbm.predict_proba(f_cur(svq, skq, Ksq, pvq))[:, 1]
        cached.append((pvq, pkq, ptq, y[teq]))
    best_d, best_s = 0.0, -1
    for d in DELTAS:
        oks, yys = [], []
        for (pvq, pkq, ptq, yq) in cached:
            oks.append(apply_rule(pvq, pkq, ptq, d) == yq); yys.append(yq)
        if not oks:
            continue
        ok = np.concatenate(oks); yy = np.concatenate(yys)
        per = [float(ok[yy == c].mean()) for c in range(C) if (yy == c).sum() > 0]
        s = float(np.mean(per))
        if s > best_s:
            best_s, best_d = s, d
    pv, pk, sv, sk, Ks = fit(tr, te, a)
    pt = gbm.predict_proba(f_cur(sv, sk, Ks, pv))[:, 1]
    preds = {'V': pv, 'K_pct': pk, 'gate': apply_rule(pv, pk, pt, 0.0),
             'd10': apply_rule(pv, pk, pt, 0.1), 'd20': apply_rule(pv, pk, pt, 0.2),
             'd30': apply_rule(pv, pk, pt, 0.3), 'srcSel': apply_rule(pv, pk, pt, best_d)}
    for nm, pr in preds.items():
        out[nm].append(ba(y[te], pr)); pp[nm][p] = out[nm][-1]
    print('%-18s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f  [src δ*=%.1f]' % (
        p, *[out[nm][-1] for nm in ARMS], best_d), flush=True)

print()
base = np.array(out['V'])
for nm in ARMS[1:]:
    v = np.array(out[nm]); d = (v - base) * 100
    print('%-8s mean %.4f  Δ %+6.2f  正港 %2d/%d  负港 %2d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), int((d < 0).sum()), d.min()))
g = np.array(out['gate'])
for nm in ['d10', 'd20', 'd30', 'srcSel']:
    v = np.array(out[nm]); dd = (v - g) * 100
    print('配对 %-7s − gate: %+6.2f pp  Wilcoxon p=%.4f' % (nm, dd.mean(), stats.wilcoxon(v, g).pvalue))
sg = [p for p in pp['V'] if p == 'Singapore']
if sg:
    print()
    print('Singapore: V %.4f  K_pct %.4f  gate %.4f  d20 %.4f  d30 %.4f  srcSel %.4f' % (
        pp['V']['Singapore'], pp['K_pct']['Singapore'], pp['gate']['Singapore'],
        pp['d20']['Singapore'], pp['d30']['Singapore'], pp['srcSel']['Singapore']))

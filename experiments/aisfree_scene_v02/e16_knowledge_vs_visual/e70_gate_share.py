"""e70: gate + a MECHANISM-DERIVED label-free feature (facility sharing), at 8 source ports.

Why this feature and not the raw knowledge: e63 showed the raw 71-dim knowledge destabilises the tail,
and e38 explained it (r=0.771: the per-instance knowledge is a port-identity carrier). e69 then found the
transferable summary: a class whose facility signature is SHARED with other classes gets harmed, and the
sharing score discriminates harm from rescue at AUC ~0.75.

Leakage discipline: the per-dim class entropy H_d is a CONSTANT estimated on the training SOURCE ports'
own labels (legitimate); the target's chips only receive the aggregation of those constants over their own
knowledge dims. No target label enters the feature. `per-dim constants from the training sources, applied to
unlabelled target chips` is the whole mechanism.

Arms: ridge_V / ridge_VK / gbm_zk / gate_old / gate_new (knowledge vectors, the failed variant) /
gate_share (one-hot + prior + sharing, no raw knowledge) / oracle3.  E70_NINNER=8 by default.
"""
import csv
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz')['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _pj in PORT_U:
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[np.ix_(_m, LEGAL)].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)

CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
ARMS = CANDS + ['gate_old', 'gate_new', 'gate_share', 'oracle3']
SUBS = int(os.environ.get('E70_SUBS', '20000'))
NINNER = int(os.environ.get('E70_NINNER', '8'))
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu, s = A.mean(0), A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def cand_preds(tr, te, sub=None):
    trs = tr if (sub is None or len(tr) <= sub) else tr[np.isin(tr, rng.choice(tr, sub, replace=False))]
    A, B = kstd(trs, te)
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[trs], y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[X[trs], A], y[trs])
    gk = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(np.c_[X[trs], A], y[trs])
    return {'ridge_V': (softmax(rv.decision_function(X[te])), rv.classes_),
            'ridge_VK': (softmax(rk.decision_function(np.c_[X[te], B])), rk.classes_),
            'gbm_zk': (gk.predict_proba(np.c_[X[te], B]), gk.classes_),
            'B': B}


def lab(P, nm):
    return P[nm][1][P[nm][0].argmax(1)]


def base_feats(P):
    fs = []
    for nm in CANDS:
        p = P[nm][0]; srt = np.sort(p, 1)
        fs.append((srt[:, -1] - srt[:, -2])[:, None])
        fs.append((-(p * np.log(p + 1e-12)).sum(1))[:, None])
    pa = [lab(P, nm) for nm in CANDS]
    for i in range(len(CANDS)):
        for j in range(i + 1, len(CANDS)):
            fs.append((pa[i] == pa[j]).astype(float)[:, None])
    return np.concatenate(fs, 1)


def share_feats(idx, Hd):
    """the mechanism feature for the given chip indices: mean and max of H_d over each chip's active dims."""
    out = np.full((len(idx), 2), np.nan)
    for r_, i in enumerate(idx):
        vs = [Hd[j] for j, d in enumerate(LEGAL) if K[i, d] != 0 and np.isfinite(Hd[j])]
        if vs:
            out[r_, 0] = float(np.mean(vs)); out[r_, 1] = float(np.max(vs))
    gm = np.nanmean(out, 0)
    out = np.where(np.isfinite(out), out, gm)
    return out


def fit_gate(F, Lc, Ft):
    ms = [HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(F, lc) for lc in Lc]
    return np.stack([m.predict_proba(Ft)[:, 1] for m in ms], 1)


out = {a: [] for a in ARMS}
dump = []
print('OUT = e70_preds.pkl | arms %s | inner %d | subs %d' % (ARMS, NINNER, SUBS), flush=True)
print('%-18s %7s %8s %7s %9s %9s %10s %8s' % ('port', 'ridge_V', 'ridge_VK', 'gbm_zk', 'gate_old', 'gate_new', 'gate_share', 'orac3'), flush=True)

for pi, p in enumerate(PORT_U):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    others = [q for q in PORT_U if q != p][:NINNER]
    # ---- per-dim class entropy from the TRAINING SOURCES' OWN labels (legitimate; no target label) ----
    Hd = np.full(len(LEGAL), np.nan)
    for j, d in enumerate(LEGAL):
        m = K[tr, d] != 0
        if m.sum() < 200:
            continue
        pp = np.bincount(y[tr][m], minlength=C).astype(float); pp /= pp.sum(); pp = pp[pp > 0]
        Hd[j] = float(-(pp * np.log(pp)).sum() / np.log(C))
    blocks = []
    for q in others:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        blocks.append((q, cand_preds(trq, teq, sub=SUBS), teq))
    if not blocks:
        for a in ARMS:
            out[a].append(float('nan'))
        continue
    Lall, Fnew, Fbase, Fshare = [], [], [], []
    for q, Pq, teq in blocks:
        lab_ = [(lab(Pq, nm) == y[teq]).astype(int) for nm in CANDS]
        Lall.append(lab_)
        nb, sh = base_feats(Pq), share_feats(teq, Hd)
        Fbase.append(nb); Fshare.append(np.c_[nb, sh])
        Fnew.append(np.c_[nb, Pq['B'], np.eye(C)[lab(Pq, 'ridge_VK')], np.eye(C)[lab(Pq, 'gbm_zk')]])
    Lc = [np.concatenate([Lall[i][j] for i in range(len(Lall))]) for j in range(len(CANDS))]
    Pt = cand_preds(tr, te)
    preds = {nm: lab(Pt, nm) for nm in CANDS}
    ok3 = np.zeros(len(te), bool)
    for nm in CANDS:
        ok3 |= (preds[nm] == y[te])
    preds['oracle3'] = np.where(ok3, y[te], -1)
    nb_t, sh_t = base_feats(Pt), share_feats(te, Hd)
    sc_old = fit_gate(np.vstack(Fbase), Lc, nb_t)
    sc_new = fit_gate(np.vstack(Fnew), Lc, np.c_[nb_t, Pt['B'], np.eye(C)[lab(Pt, 'ridge_VK')], np.eye(C)[lab(Pt, 'gbm_zk')]])
    sc_shr = fit_gate(np.vstack(Fshare), Lc, np.c_[nb_t, sh_t])
    preds['gate_old'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(sc_old.argmax(1))])
    preds['gate_new'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(sc_new.argmax(1))])
    preds['gate_share'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(sc_shr.argmax(1))])
    for a in ARMS:
        out[a].append(ba(y[te], preds[a]))
    if pi == 0:
        print('  对齐自检 y|V VK GBM|old new share:', flush=True)
        for i in range(5):
            print('    %d|%d %d %d|%d %d %d' % (y[te][i], preds['ridge_V'][i], preds['ridge_VK'][i], preds['gbm_zk'][i],
                                                preds['gate_old'][i], preds['gate_new'][i], preds['gate_share'][i]), flush=True)
        print('  H_d 可用维 %d 中位 %.3f' % (int(np.isfinite(Hd).sum()), np.nanmedian(Hd)), flush=True)
    dump.append({'port': p, 'y': y[te].copy(), 'pred': {nm: preds[nm].copy() for nm in CANDS},
                 'sc_old': sc_old, 'sc_new': sc_new, 'sc_shr': sc_shr, 'share': sh_t, 'Hd': Hd.copy()})
    print('%-18s %7.3f %8.3f %7.3f %9.3f %9.3f %10.3f %8.3f' % (p, *[out[a][-1] for a in ARMS]), flush=True)

pickle.dump(dump, open('e70_preds.pkl', 'wb'))
print()
for a in ARMS:
    v = np.array(out[a]); d = (v - np.array(out['ridge_V'])) * 100
    print('%-14s mean %.4f  Δ vs ridge_V %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        a, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
print()
for a, b in [('gate_share', 'gate_old'), ('gate_share', 'gate_new'), ('gate_old', 'gate_new')]:
    x, z = np.array(out[a]), np.array(out[b])
    m = ~(np.isnan(x) | np.isnan(z))
    print('配对 %s − %s: %+6.2f pp  p=%.4f' % (a, b, (x[m] - z[m]).mean() * 100, stats.wilcoxon(x[m], z[m]).pvalue))

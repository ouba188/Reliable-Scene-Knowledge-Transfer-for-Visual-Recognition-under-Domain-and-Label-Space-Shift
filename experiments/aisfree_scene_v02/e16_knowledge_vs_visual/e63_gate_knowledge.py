"""e63 = (1) gate features += the knowledge vector + predicted-class one-hots + a source-estimated per-class harm prior,
        (3) a conservative margin theta SELECTED ON THE SOURCE (nested leave-one-inner-port-out),
        (2) source-port count via E63_NINNER (3 default, 8 = the autonomous-source run) + per-source-port logging.

Control arm keeps the OLD feature set so any change is attributable. Alignment spot-check on fold 1 once
(margin/entropy/pick/out alignment is the fault class that cost two runs in e60 -- see e60dbg.py).
ponytail: the per-class prior is a global aggregate over the inner episodes (not re-nested per q); its
influence is a 2-dim feature out of ~95 and the aggregate is over thousands of items, so the leak is tiny.
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
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _pj in PORT_U:
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)

CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
ARMS = CANDS + ['gate_old', 'gate_new', 'gate_new_theta', 'gate_nk', 'gate_nk_theta', 'oracle3']
SUBS = int(os.environ.get('E63_SUBS', '20000'))
NINNER = int(os.environ.get('E63_NINNER', '3'))
THETAS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
TAG = os.environ.get('E63_TAG', 'n%d' % (NINNER + 100 * (SUBS // 100000)))
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
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


def new_feats(P, prior, with_know=True):
    """with_know=False => drop the raw 71-dim knowledge (e38: its per-instance content is port-identity
    laden, corr=0.771 with I/port, so it lets the gate memorise sources). Keep only the transferable part:
    predicted-class one-hots + the source-estimated per-class harm prior."""
    cls_v = lab(P, 'ridge_V')
    pr = np.array([prior.get(int(c), (0.0, 0.0)) for c in cls_v], dtype=np.float32)
    parts = [base_feats(P)]
    if with_know:
        parts.append(P['B'])
    parts += [np.eye(C)[lab(P, 'ridge_VK')], np.eye(C)[lab(P, 'gbm_zk')], pr]
    return np.concatenate(parts, 1)


def fit_gate(F, Lc, Ft):
    ms = [HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(F, lc) for lc in Lc]
    return np.stack([m.predict_proba(Ft)[:, 1] for m in ms], 1)


out = {a: [] for a in ARMS}
src_log, dump = [], []
print('OUT = e63_preds_%s.pkl' % TAG)
print('cands %s  arms %s  inner %d  subs %d' % (CANDS, ARMS, NINNER, SUBS), flush=True)
print('%-18s %7s %8s %7s %9s %9s %9s %9s %9s %8s' % ('port', 'ridge_V', 'ridge_VK', 'gbm_zk',
                                             'gate_old', 'gate_new', 'gateNewTh', 'gate_nk', 'gateNkTh', 'orac3'), flush=True)

for pi, p in enumerate(PORT_U):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    others = [q for q in PORT_U if q != p][:NINNER]
    blocks = []                                    # [(q, Pq, Lq)] per inner source port
    for q in others:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        Pq = cand_preds(trq, teq, sub=SUBS)
        blocks.append((q, Pq, [(lab(Pq, nm) == y[teq]).astype(int) for nm in CANDS]))
    if not blocks:
        for a in ARMS:
            out[a].append(float('nan'))
        continue
    # ---- per-class harm prior, keyed by the PREDICTED class (ridge_V) of the inner episodes ----
    cnt = np.zeros((C, len(CANDS))); tot = np.zeros(C)
    for q, Pq, Lq in blocks:
        cl = lab(Pq, 'ridge_V')
        for i in range(len(cl)):
            tot[cl[i]] += 1
            for j in range(len(CANDS)):
                cnt[cl[i], j] += Lq[j][i]
    acc = cnt / np.maximum(tot[:, None], 1)
    prior = {c: (float(acc[c, 1] - acc[c, 0]), float(acc[c, 2] - acc[c, 0])) for c in range(C) if tot[c] >= 20}
    F_all = np.vstack([new_feats(Pq, prior) for q, Pq, _ in blocks])
    Lall = [np.concatenate([Lq[j] for _, _, Lq in blocks]) for j in range(len(CANDS))]
    # ---- nested source-side evaluation + theta selection ----
    qtbl = {}
    for qi, (q, Pq, Lq) in enumerate(blocks):
        Fq = new_feats(Pq, prior)
        Fother = np.vstack([new_feats(P2, prior) for j, (q2, P2, _) in enumerate(blocks) if j != qi])
        Lother = [np.concatenate([L2[j] for j2, (q2, P2, L2) in enumerate(blocks) if j2 != qi])
                  for j in range(len(CANDS))]
        if len(Fother) < 50:
            continue
        scq = fit_gate(Fother, Lother, Fq)
        pq = {nm: lab(Pq, nm) for nm in CANDS}
        qy = y[np.where(ports == q)[0]]
        qtbl[q] = {'y': qy, 'pred': pq, 'sc': scq,
                   'ba_v': ba(qy, pq['ridge_V']), 'ba_vk': ba(qy, pq['ridge_VK']), 'ba_gk': ba(qy, pq['gbm_zk']),
                   'gate': ba(qy, np.array([pq[CANDS[j]][i] for i, j in enumerate(scq.argmax(1))]))}
    # theta* = the source-side argmax of the WORST inner-port gain (tail-targeted, no target labels)
    best_th, best_val = 0.0, -9e9
    for th in THETAS:
        vals = []
        for q, d in qtbl.items():
            pick = np.where(d['sc'][:, 2] - d['sc'][:, 1] > th, 2, 1)
            pv = np.array([d['pred'][CANDS[j]][i] for i, j in enumerate(pick)])
            vals.append((ba(d['y'], pv) - d['ba_v']) * 100)
        v = min(vals) if vals else -9e9
        if v > best_val:
            best_th, best_val = th, v
    src_log.append({'port': p, 'theta': best_th, 'src_worst_gain': best_val,
                    'per_src': {q: {kk: d[kk] for kk in ('ba_v', 'ba_vk', 'ba_gk', 'gate')} for q, d in qtbl.items()}})
    # ---- target models + arms ----
    Pt = cand_preds(tr, te)
    preds = {nm: lab(Pt, nm) for nm in CANDS}
    ok3 = np.zeros(len(te), bool)
    for nm in CANDS:
        ok3 |= (preds[nm] == y[te])
    preds['oracle3'] = np.where(ok3, y[te], -1)
    Fo_t, Fn_t = base_feats(Pt), new_feats(Pt, prior)
    sc_old = fit_gate(np.vstack([base_feats(Pq) for _, Pq, _ in blocks]), Lall, Fo_t)
    sc_new = fit_gate(F_all, Lall, Fn_t)
    preds['gate_old'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(sc_old.argmax(1))])
    preds['gate_new'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(sc_new.argmax(1))])
    pick = np.where(sc_new[:, 2] - sc_new[:, 1] > best_th, 2, 1)       # the source-selected theta rule
    preds['gate_new_theta'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(pick)])
    sc_nk = fit_gate(np.vstack([new_feats(Pq, prior, with_know=False) for _, Pq, _ in blocks]), Lall,
                     new_feats(Pt, prior, with_know=False))
    preds['gate_nk'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(sc_nk.argmax(1))])
    pick_nk = np.where(sc_nk[:, 2] - sc_nk[:, 1] > best_th, 2, 1)
    preds['gate_nk_theta'] = np.array([preds[CANDS[j]][i] for i, j in enumerate(pick_nk)])
    for nm in ARMS:
        out[nm].append(ba(y[te], preds[nm]))
    if pi == 0:
        print('  [对齐自检] y | ridge_V ridge_VK gbm_zk | gate_old : gate_new', flush=True)
        for i in range(6):
            print('    %d | %d %d %d | %d : %d' % (y[te][i], preds['ridge_V'][i], preds['ridge_VK'][i],
                                                   preds['gbm_zk'][i], preds['gate_old'][i], preds['gate_new'][i]), flush=True)
        print('  源港 theta* = %.2f (最差内层港增益 %+.2f)' % (best_th, best_val), flush=True)
    dump.append({'port': p, 'y': y[te].copy(), 'pred': {nm: preds[nm].copy() for nm in CANDS},
                 'sc_old': sc_old, 'sc_new': sc_new, 'theta': best_th})
    _v=[out[a][-1] for a in ARMS]
    print('%-18s %7.3f %8.3f %7.3f %9.3f %9.3f %9.3f %9.3f %9.3f %8.3f' % (p, *_v), flush=True)

pickle.dump({'dump': dump, 'src_log': src_log}, open('e63_preds_%s.pkl' % TAG, 'wb'))
print()
for nm in ARMS:
    v = np.array(out[nm]); d = (v - np.array(out['ridge_V'])) * 100
    print('%-14s mean %.4f  Δ vs ridge_V %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
print()
for a, b in [('gate_new', 'gate_old'), ('gate_nk', 'gate_old'), ('gate_nk', 'gate_new'), ('gate_nk_theta', 'gate_nk')]:
    x, z = np.array(out[a]), np.array(out[b])
    m = ~(np.isnan(x) | np.isnan(z))
    print('配对 %s − %s: %+6.2f pp  p=%.4f' % (a, b, (x[m] - z[m]).mean() * 100, stats.wilcoxon(x[m], z[m]).pvalue))

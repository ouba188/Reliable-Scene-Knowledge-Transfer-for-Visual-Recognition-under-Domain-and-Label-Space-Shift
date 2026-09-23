"""e59: redo the gate + class-aware threshold under the STRONG judge (GBM).

Base pair: gbm_z (features only) vs gbm_zk (features + within-port percentile knowledge).
The gate now decides per instance between gbm_z and gbm_zk (the honest fallback is the
knowledge-free model, not the ridge). Class-aware threshold kept (higher bar to flip into a rarer
class). Everything is selected on source ports only.

Cost control: the G-training models (inner source ports) are trained on a 20k subsample; only the
final target models use all data.

Reference (e57, no gating): gbm_z 0.4531 | gbm_zk 0.5272 | worst -15.30
"""
import csv, os
from pathlib import Path

import numpy as np
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
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
DELTAS = [0.0, 0.1, 0.2, 0.3]
ARMS = ['gbm_z', 'gbm_zk', 'gated', 'gated_d20', 'srcsel']
SUBS = int(os.environ.get('E59_SUBS', '20000'))
NINNER = 3
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def mk():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0)


def fit_pair(tr, te, sub=None):
    A, B = kstd(tr, te)
    trs = tr if (sub is None or len(tr) <= sub) else tr[np.isin(tr, rng.choice(tr, sub, replace=False))]
    As, Bs = kstd(trs, te)
    mz = mk().fit(X[trs], y[trs])
    mzk = mk().fit(np.c_[X[trs], As], y[trs])
    pz = mz.predict_proba(X[te])
    pzk = mzk.predict_proba(np.c_[X[te], B])
    return pz, pzk, B


def feats(pz, pzk, B, pv):
    srt = np.sort(pz, 1); marg = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(pz * np.log(pz + 1e-12)).sum(1))[:, None]
    srtk = np.sort(pzk, 1); mk_ = (srtk[:, -1] - srtk[:, -2])[:, None]
    dis = (pzk.argmax(1) != pv).astype(float)[:, None]
    oh = np.zeros((len(pv), C - 1)); oh[np.arange(len(pv)), np.minimum(pv, C - 2)] = 1
    return np.c_[marg, ent, mk_, dis, oh, B]


def rarity(pred):
    cnt = np.bincount(pred, minlength=C).astype(float); cnt[cnt == 0] = 1.0
    w = 1.0 / cnt
    return w / w.mean()


def apply_rule(pv, pk, pt, rc, tau, d):
    r_k = np.array([rc[c] for c in pk])
    return np.where((pt > tau + d * r_k) | (pk == pv), pk, pv)


print('%-18s %8s %8s %8s %8s %8s' % tuple(['port'] + ARMS))
print('-' * 68)
out = {a: [] for a in ARMS}
for p in PORT_U:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    # --- G training on inner source ports (nested LOO, subsampled) ---
    F1, G, rec = [], [], []
    for q in [x for x in PORT_U if x != p][:NINNER]:
        trq = np.where((ports != p) & (ports != q))[0]; teq = np.where(ports == q)[0]
        if len(trq) < 200:
            continue
        pz, pzk, B = fit_pair(trq, teq, sub=SUBS)
        pvq = pz.argmax(1); pkq = pzk.argmax(1)
        g = np.zeros(len(teq)); g[(pvq != y[teq]) & (pkq == y[teq])] = 1; g[(pvq == y[teq]) & (pkq != y[teq])] = -1
        m = g != 0
        if m.sum() < 20:
            continue
        F1.append(feats(pz[m], pzk[m], B[m], pvq[m])); G.append(g[m])
        rec.append((pz, pzk, B, pvq, pkq, teq))
    pz, pzk, B = fit_pair(tr, te)
    pv = pz.argmax(1); pk = pzk.argmax(1)
    res = {'gbm_z': ba(y[te], pv), 'gbm_zk': ba(y[te], pk)}
    if G:
        gbm = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0).fit(
            np.vstack(F1), (np.concatenate(G) > 0).astype(int))
        # tau on source (pooled gated BA), delta by fixed grid -> srcsel picks delta
        pt_src, yv, pvs, pks = [], [], [], []
        rc_src = rarity(np.concatenate([r[3] for r in rec]))
        for (pzq, pzkq, Bq, pvq, pkq, teq) in rec:
            pt_src.append(gbm.predict_proba(feats(pzq, pzkq, Bq, pvq))[:, 1])
            yv.append(y[teq]); pvs.append(pvq); pks.append(pkq)
        pt_src = np.concatenate(pt_src); yv = np.concatenate(yv)
        pvs = np.concatenate(pvs); pks = np.concatenate(pks)
        best = (-1, 0.5, 0.0)
        for t in TAUS:
            for d in DELTAS:
                pr = apply_rule(pvs, pks, pt_src, rc_src, t, d)
                s = ba(yv, pr)
                if s > best[0]:
                    best = (s, t, d)
        _, tau, dsel = best
        pt = gbm.predict_proba(feats(pz, pzk, B, pv))[:, 1]
        rc = rarity(pv)
        res['gated'] = ba(y[te], apply_rule(pv, pk, pt, rc, 0.5, 0.0))
        res['gated_d20'] = ba(y[te], apply_rule(pv, pk, pt, rc, 0.5, 0.2))
        res['srcsel'] = ba(y[te], apply_rule(pv, pk, pt, rc, tau, dsel))
        res['_src'] = (tau, dsel)
    else:
        res['gated'] = res['gbm_zk']; res['gated_d20'] = res['gbm_zk']; res['srcsel'] = res['gbm_zk']
        res['_src'] = (0.5, 0.0)
    for nm in ARMS:
        out[nm].append(res[nm])
    print('%-18s %8.3f %8.3f %8.3f %8.3f %8.3f  [src τ=%.2f δ=%.1f]' % (
        p, res['gbm_z'], res['gbm_zk'], res['gated'], res['gated_d20'], res['srcsel'],
        res['_src'][0], res['_src'][1]), flush=True)

print()
for nm in ARMS:
    v = np.array(out[nm]); d = (v - np.array(out['gbm_z'])) * 100
    print('%-10s mean %.4f  Δ vs gbm_z %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
gz = np.array(out['gbm_zk'])
for nm in ['gated', 'gated_d20', 'srcsel']:
    v = np.array(out[nm]); dd = (v - gz) * 100
    try:
        pw = stats.wilcoxon(v, gz).pvalue
    except Exception:
        pw = float('nan')
    print('配对 %-10s − gbm_zk: %+6.2f pp  Wilcoxon p=%.4f' % (nm, dd.mean(), pw))

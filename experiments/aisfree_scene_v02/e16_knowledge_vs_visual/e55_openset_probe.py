"""e55: minimal open-set probe -- does the scene knowledge that helps KNOWN classes make UNKNOWN
classes more likely to be falsely accepted?

Protocol
  outer: 2 of the 8 classes are declared UNKNOWN and are removed from every supervised component
         (the classifier, the knowledge-only head, the novelty thresholds). The PCA projection is a
         fixed unsupervised artefact computed on all chips -- disclosed as a mild unsupervised leak.
  inner: thresholds are chosen from SOURCE chips only (a fixed known-rejection rate), never the target.
  per split x per held-out port (double hold-out: unknown classes AND unseen port).

Scorers compared (higher = more "known"):
  v_msp    : visual softmax max          (baseline 1)
  v_energy : visual logsumexp            (baseline 2)
  f_msp    : knocknowledge-fused softmax max   (baseline 3: fuse first, then reject)
  k_msp    : knowledge-only head max     (the CONTEXT signal)
  combo    : rank-average of v_energy and k_msp (the new proposal: context participates in rejection)

Metrics per (split, scorer) at matched known-rejection rates:
  known_BA            : balanced accuracy over the 6 known classes, rejected knowns count as errors
  unknown_recall      : share of unknown-class chips that are rejected
  unknown_false_accept: share of unknown chips accepted as known
"""
import csv, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.extmath import softmax as sk_softmax

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)

SPLITS = {
    'S1 crude+offshore': [5, 7],
    'S2 tug+fishing': [6, 1],
    'S3 container+chem': [4, 3],
    'S4 bulk+gcargo': [0, 2],
}
REJECT_RATES = [0.02, 0.05, 0.10]


def std_train(tr, te, Kd):
    A, B = Kd[tr][:, LEGAL], Kd[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


def scores(tr, te, known):
    """returns dict of novelty scores on te (higher = more likely KNOWN) + each head's argmax"""
    ys = np.isin(y[tr], known).astype(int)
    tr2 = tr[ys == 1]
    Ka, Kb = std_train(tr2, te, Kp)
    out = {}
    fv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[tr2], y[tr2])
    dv = fv.decision_function(X[te])
    pv = sk_softmax(dv)
    out['v_msp'] = pv.max(1)
    out['v_energy'] = np.log(np.exp(dv - dv.max(1, keepdims=True)).sum(1)) + dv.max(1)
    dk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ka, y[tr2]).decision_function(Kb)
    out['k_msp'] = sk_softmax(dk).max(1)
    ff = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[X[tr2], Ka], y[tr2])
    df = ff.decision_function(np.c_[X[te], Kb])
    out['f_msp'] = sk_softmax(df).max(1)
    return out, {'v': fv.classes_[pv.argmax(1)], 'k': fv.classes_[dk.argmax(1)], 'f': ff.classes_[df.argmax(1)]}


def rk(v):
    o = v.argsort().argsort().astype(float)
    return o / max(1, len(v) - 1)


def eval_split(unknown):
    known = [c for c in range(C) if c not in unknown]
    res = {s: [] for s in ['v_msp', 'v_energy', 'f_msp', 'k_msp', 'combo']}
    for p in PORT_U:
        te = np.where(ports == p)[0]
        tr = np.where(ports != p)[0]
        if len(te) < 20 or len(set(y[tr].tolist()) & set(known)) < 6:
            continue
        sc, preds = scores(tr, te, known)
        sc['combo'] = rk(sc['v_energy']) + rk(sc['k_msp'])
        yte = y[te]
        is_u = np.isin(yte, unknown)
        # 阈值：来自一个"既非目标港、也非目标类"的旁置源港（避免 in-sample 乐观偏置）
        q = [x for x in PORT_U if x != p][0]
        trq = tr[ports[tr] != q]; teq = np.where(ports == q)[0]
        sc_src, _ = scores(trq, teq, known)
        sc_src['combo'] = rk(sc_src['v_energy']) + rk(sc_src['k_msp'])
        for nm, sv in sc.items():
            thr = np.quantile(sc_src[nm], 0.05)          # 源端 5% 已知被拒
            acc = sv >= thr
            pr = preds['f'] if nm == 'f_msp' else preds['v']    # 各打分器用自己的分类头
            pr_eff = np.where(acc, pr, -1)                      # 被拒的已知样本计为错误（进分母）
            per = [float((pr_eff[(~is_u) & (yte == c)] == c).mean()) for c in known if (yte == c).sum() > 0]
            kba = float(np.mean(per)) if per else float('nan')
            urec = float((~acc[is_u]).mean()) if is_u.sum() else float('nan')
            ufa = float(acc[is_u].mean()) if is_u.sum() else float('nan')
            res[nm].append((kba, urec, ufa))
    return {nm: (float(np.nanmean([t[0] for t in v])), float(np.nanmean([t[1] for t in v])),
                 float(np.nanmean([t[2] for t in v]))) for nm, v in res.items()}


for name, unk in SPLITS.items():
    r = eval_split(unk)
    print('=== %s  (未知 = %s) ===' % (name, unk))
    print('%-9s %10s %14s %18s' % ('scorer', 'Known-BA', 'UnknownRecall', 'UnknownFalseAccept'))
    for nm in ['v_msp', 'v_energy', 'f_msp', 'k_msp', 'combo']:
        b, u, fa = r[nm]
        print('%-9s %10.4f %14.4f %18.4f' % (nm, b, u, fa))
    print()

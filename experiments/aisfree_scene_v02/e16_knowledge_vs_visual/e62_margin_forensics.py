"""e62: conservative-margin selection + per-port/per-class forensics, from e61's dumped decisions.

Reads e61_preds.pkl (per port: truth, each candidate's labels, the selector's 3 scores, the picks).
No refitting -- the whole thing is arithmetic on decisions already made.

Two label-free ways to be conservative:
  A) absolute margin: use the strong judge only when score[gbm_zk] - score[ridge_VK] > theta.
  B) quantile rule: switch to the strong judge only on the top (1-q) fraction of that difference,
     i.e. theta = quantile_q(diff) computed on the target port itself (still label-free).

Also prints the per-port/per-class forensic for the worst ports: which classes the strong judge
wins and loses, and the top confusion flips the knowledge causes.
"""
import pickle
from collections import Counter

import numpy as np
from scipy import stats

C = 8
CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
# pickle is safe here and required: e61_preds.pkl is written by our own e61_ceiling.py on this
# machine (dicts of numpy arrays, variable-length per port); nothing external is ever loaded.
dump = pickle.load(open('e61_preds.pkl', 'rb'))
print('ports dumped:', len(dump))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def run_rule(fn):
    """fn(scores, dl) -> index into CANDS per instance."""
    out, npick = {}, []
    for d in dump:
        sc = d['score']; pick = fn(sc)
        npick.append(float((pick == 2).mean()))
        out[d['port']] = ba(d['y'], np.array([d['pred'][CANDS[p]][j] for j, p in enumerate(pick)]))
    return out, float(np.mean(npick))


base = {p: ba(d['y'], d['pred']['ridge_V']) for d in dump for p in [d['port']]}
grid = {}
for th in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]:
    grid['theta=%.2f' % th] = run_rule(lambda sc, t=th: np.where(sc[:, 2] - sc[:, 1] > t, 2, 1))
for q in [0.5, 0.7, 0.9, 0.97]:
    grid['quantile=%.2f' % q] = run_rule(lambda sc, qq=q: np.where(sc[:, 2] - sc[:, 1] > np.quantile(sc[:, 2] - sc[:, 1], qq), 2, 1))
grid['always strong'] = run_rule(lambda sc: np.full(len(sc), 2))
grid['always safe'] = run_rule(lambda sc: np.full(len(sc), 1))
grid['argmax(3)'] = run_rule(lambda sc: sc.argmax(1))

print()
print('%-16s %8s %8s %7s %8s' % ('rule', 'mean', 'Δ base', 'pos/24', 'worst'))
for nm, (o, fr) in grid.items():
    v = np.array([o[p] for p in sorted(o)]); b0 = np.array([base[p] for p in sorted(o)])
    d = (v - b0) * 100
    print('%-16s %8.4f %+8.2f %7d %+8.2f   (强判据占比 %.0f%%)' % (
        nm, v.mean(), d.mean(), int((d > 0).sum()), d.min(), fr * 100))

# ---- per-port / per-class forensics on the resulting worst ports
print()
print('=' * 78)
o_best, fr_best = grid['quantile=0.90']
worst = sorted(o_best, key=lambda p: o_best[p] - base[p])[:3]
print('保守边距后仍最差的港:', ['%s %+.2f' % (p, (o_best[p] - base[p]) * 100) for p in worst])
for p in worst:
    d = [x for x in dump if x['port'] == p][0]
    y, pred = d['y'], d['pred']
    print()
    print('--- %s ---' % p)
    print('%6s %5s %9s %9s %9s %9s' % ('class', 'n', 'ridge_V', 'ridge_VK', 'gbm_zk', 'gate3'))
    for c in range(C):
        m = (y == c)
        if not m.any():
            continue
        print('%6d %5d %9.3f %9.3f %9.3f %9.3f' % (
            c, m.sum(), (pred['ridge_V'][m] == c).mean(), (pred['ridge_VK'][m] == c).mean(),
            (pred['gbm_zk'][m] == c).mean(),
            (np.array([pred[CANDS[i]][j] for j, i in enumerate(d['pick3'])])[m] == c).mean()))
    # the flips the knowledge causes: true class -> predicted class, VK vs V
    v, k = pred['ridge_V'], pred['ridge_VK']
    gain = Counter(zip(y[(v != y) & (k == y)], k[(v != y) & (k == y)]))
    harm = Counter(zip(y[(v == y) & (k != y)], k[(v == y) & (k != y)]))
    print('  救回 top:', ['%d→%d x%d' % (a, b, n) for (a, b), n in gain.most_common(4)], '共', sum(gain.values()))
    print('  误伤 top:', ['%d→%d x%d' % (a, b, n) for (a, b), n in harm.most_common(4)], '共', sum(harm.values()))

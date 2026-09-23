"""e66: within-port class-level view -- support, composition, and the noise floor.

The port BA is an unweighted mean over classes, so a port's headline number can be driven by a single
tiny class (n=1..4 cells exist in this grid) and, conversely, a genuine per-class loss can be hidden by a
dominant class. This computes, per port:
  n / composition skew, the BA delta (unweighted), the ACCURACY delta (composition-weighted),
  the support-filtered BA delta (classes with n>=20 only), and the per-cell binomial SE floor.
Plus the per-class port-consistency (in how many ports is this class's delta positive).
Arm: gate_old with 8 source ports vs ridge_V (the recommended configuration).
"""
import pickle
from collections import defaultdict

import numpy as np

C = 8
CANDS = ['ridge_V', 'ridge_VK', 'gbm_zk']
NAMES = {0: 'bulk_carrier', 1: 'fishing_vessel', 2: 'general_cargo', 3: 'product_chem_tanker',
         4: 'container_ship', 5: 'crude_oil_tanker', 6: 'tug_towing', 7: 'offshore_supply'}
d = pickle.load(open('e63_preds_n8.pkl', 'rb'))          # our own artifact from e63
recs = []
for rec in d['dump']:
    y, P = rec['y'], rec['pred']
    s_old = rec['sc_old']
    g = np.array([P[CANDS[j]][i] for i, j in enumerate(s_old.argmax(1))])
    recs.append({'port': rec['port'], 'y': y, 'v': P['ridge_V'], 'g': g})


def rc(y, p, c):
    m = (y == c)
    return (float((p[m] == c).mean()), int(m.sum())) if m.any() else (float('nan'), 0)


cells = [(r['port'], c, *rc(r['y'], r['v'], c)[1:], rc(r['y'], r['v'], c)[0], rc(r['y'], r['g'], c)[0], rc(r['y'], r['g'], c)[1])
         for r in recs for c in range(C) if (r['y'] == c).any()]
print('网格单元数 %d' % len(cells))
for thr in (1, 5, 20, 50, 100):
    m = [x for x in cells if x[2] < thr]
    print('  n < %-4d 的单元: %3d (%.0f%%)  其中 n<5 的类别占样本量 %.1f%%' % (
        thr, len(m), 100.0 * len(m) / len(cells),
        100.0 * sum(x[2] for x in m) / sum(x[2] for x in cells)))
print()
print('逐格 n 的分位: p10=%d p25=%d 中位=%d p75=%d p90=%d' % tuple(
    int(np.percentile([x[2] for x in cells], q)) for q in (10, 25, 50, 75, 90)))

print()
print('%-18s %5s %6s %8s %8s %9s %9s %7s' % ('port', 'n', '类别数', 'BA Δ', 'ACC Δ', 'BAΔ(n≥20)', '最差类', '主类占比'))
rows = []
for r in recs:
    y = r['y']
    cs = [c for c in range(C) if (y == c).any()]
    ns = {c: int((y == c).sum()) for c in cs}
    ba_v = np.mean([rc(y, r['v'], c)[0] for c in cs]); ba_g = np.mean([rc(y, r['g'], c)[0] for c in cs])
    acc_v = float((r['v'] == y).mean()); acc_g = float((r['g'] == y).mean())
    big = [c for c in cs if ns[c] >= 20]
    ba20 = (np.mean([rc(y, r['g'], c)[0] for c in big]) - np.mean([rc(y, r['v'], c)[0] for c in big])) if big else float('nan')
    worst = min(cs, key=lambda c: rc(y, r['g'], c)[0] - rc(y, r['v'], c)[0])
    rows.append((r['port'], len(y), len(cs), (ba_g - ba_v) * 100, (acc_g - acc_v) * 100, ba20 * 100,
                 NAMES[worst], max(ns.values()) / len(y)))
for x in sorted(rows, key=lambda z: z[3]):
    print('%-18s %5d %6d %+8.2f %+8.2f %+9.2f %9s %6.0f%%' % (x[0], x[1], x[2], x[3], x[4], x[5], x[6][:9], x[7] * 100))
sign = [x for x in rows if np.sign(x[3]) != np.sign(x[4])]
print()
print('BA 与 ACC 符号不一致的港: %d 个 ->' % len(sign), [(x[0], round(x[3], 1), round(x[4], 1)) for x in sign])
print('支持过滤(n>=20)后符号翻转的港: %d 个 ->' % sum(1 for x in rows if np.sign(x[3]) != np.sign(x[5])),
      [(x[0], round(x[3], 1), round(x[5], 1)) for x in rows if np.sign(x[3]) != np.sign(x[5])])

print()
print('按类别看"在多少个港为正"（gate_old − ridge_V，要求该港该类 n>=20）:')
for c in range(C):
    vs = [(rc(y, r['g'], c)[0] - rc(y, r['v'], c)[0]) for r in recs for y in [r['y']]
          if (y == c).sum() >= 20]
    if not vs:
        continue
    a = np.array(vs)
    print('  %-20s 港数 %2d  正 %2d/%2d  均值 %+.1fpp  最差 %+.1f  最好 %+.1f' % (
        NAMES[c], len(a), int((a > 0).sum()), len(a), a.mean() * 100, a.min() * 100, a.max() * 100))

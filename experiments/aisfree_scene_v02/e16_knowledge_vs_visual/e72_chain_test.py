"""e72: the sharp mechanism test -- liquid-chain vs dry-chain context.

The authoritative layout (relation_features.json) names the groups, so the hypothesis can be tested with
semantic labels instead of the entropy proxy of e69:
  liquid_chain = dims {42,43,44,57}   (liquid berth chain: joint / weakest / imbalance / relative evidence)
  dry_chain    = dims {45,46,47,58}
  general_logistics = {48,49,50,59}   liquid_industrial = {51,52,53}   navigation_interface = {54,55,56}
Hypothesis: chips whose context is LIQUID get harmed by the knowledge (crude/chemical/offshore share the
liquid berths), chips whose context is DRY get rescued (bulk berths are class-exclusive).

All in the label-free within-port percentile coordinate the models actually see. Outcomes from the frozen
e61 predictions. Reported: harm-vs-rescue AUC per group score, the liquid-minus-dry contrast, quartile tables,
the per-true-class sanity check, and -- the practical payoff -- a one-line mechanism-derived rule
('use the knowledge arm only when the context is dry') scored on the same folds.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
NAMES = {0: 'bulk', 1: 'fish', 2: 'gcargo', 3: 'pchem', 4: 'cont', 5: 'crude', 6: 'tug', 7: 'offsh'}
GROUPS = {
    'liquid_chain': [42, 43, 44, 57],
    'dry_chain': [45, 46, 47, 58],
    'general_logistics': [48, 49, 50, 59],
    'liquid_industrial': [51, 52, 53],
    'nav_interface': [54, 55, 56],
}

Kp = np.zeros((len(y), len(LEGAL)))
for pj in sorted(set(ports.tolist())):
    m = ports == pj
    Kp[np.ix_(m, range(len(LEGAL)))] = K[np.ix_(m, LEGAL)].argsort(0).argsort(0) / max(1, int(m.sum()) - 1)

score = {g: Kp[:, ds].mean(1) for g, ds in GROUPS.items()}
score['liquid_minus_dry'] = score['liquid_chain'] - score['dry_chain']

d61 = pickle.load(open('e61_preds.pkl', 'rb'))     # our own artifact
resc = np.zeros(len(y), bool); harm = np.zeros(len(y), bool)
Vpred = np.empty(len(y), int); VKpred = np.empty(len(y), int)
for rec in d61:
    sel = np.where(ports == rec['port'])[0]
    v, vk, yy = rec['pred']['ridge_V'], rec['pred']['ridge_VK'], rec['y']
    Vpred[sel] = v; VKpred[sel] = vk
    resc[sel] = (v != yy) & (vk == yy)
    harm[sel] = (v == yy) & (vk != yy)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


sel = harm | resc
print('非中性 chip %d（救回 %d / 误伤 %d）' % (sel.sum(), resc.sum(), harm.sum()))
print()
print('%-18s %10s %10s %12s' % ('score', 'AUC(rescue)', 'AUC(harm)', 'AUC(discrim.)'))
for g in list(GROUPS) + ['liquid_minus_dry']:
    s = score[g][sel]
    a_r = roc_auc_score(resc[sel], s); a_h = roc_auc_score(harm[sel], s)
    print('%-18s %10.3f %10.3f %12.3f' % (g, a_r, a_h, a_h - a_r + 0.5))

print()
c = score['liquid_minus_dry']
q1, q3 = np.quantile(c[sel], [0.25, 0.75])
lo = sel & (c <= q1); hi = sel & (c >= q3)
for nm, m in [('最干（liquid-dry 最低四分位）', lo), ('最液（最高四分位）', hi)]:
    print('%-28s n=%4d  误伤率 %.3f  救回率 %.3f' % (nm, m.sum(), harm[m].sum() / m.sum(), resc[m].sum() / m.sum()))

print()
print('逐类 sanity（该类的 liquid_chain / dry_chain 平均分位）:')
for cl in range(C):
    m = y == cl
    if m.sum() < 20:
        continue
    print('  %-7s n=%5d  liquid %.3f  dry %.3f  Δ=%.3f ｜ ΔVK %+5.1fpp' % (
        NAMES[cl], m.sum(), score['liquid_chain'][m].mean(), score['dry_chain'][m].mean(),
        (score['liquid_chain'][m].mean() - score['dry_chain'][m].mean()),
        100 * ((VKpred[y == cl] == cl).mean() - (Vpred[y == cl] == cl).mean())))

print()
print('机制导出的简单规则（不用闸门、不训练）:')
rules = {
    'd=0.5 干才用': score['liquid_minus_dry'] < 0.5,
    'd=0.0 干才用': score['liquid_minus_dry'] < 0.0,
    'd=-0.5 干才用': score['liquid_minus_dry'] < -0.5,
}
for nm, use_k in rules.items():
    pred = np.where(use_k, VKpred, Vpred)
    per = []
    for pj in sorted(set(ports.tolist())):
        m = ports == pj
        per.append(ba(y[m], pred[m]))
    per = np.array(per); base = []
    for pj in sorted(set(ports.tolist())):
        m = ports == pj
        base.append(ba(y[m], Vpred[m]))
    d = (per - np.array(base)) * 100
    print('  %-14s 平均 %.4f  Δ %+6.2f  逐港正 %2d/24  worst %+6.2f' % (
        nm, per.mean(), d.mean(), int((d > 0).sum()), d.min()))

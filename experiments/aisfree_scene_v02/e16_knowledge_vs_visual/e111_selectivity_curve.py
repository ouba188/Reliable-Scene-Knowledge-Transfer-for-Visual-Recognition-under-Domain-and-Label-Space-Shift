"""e111: the selectivity-regret curve -- turning the mechanism into a quantitative, testable relation.

The open question a top-venue referee will ask is not "how deep is the network" but "what does the mechanism
predict, and does the prediction hold?". Tonight's numbers already contain the ingredients:
  * a per-instance selector over two candidates (visual V, visual+knowledge VK) whose sign is predictable at
    AUC ~0.72-0.75 from deployment-legal features;
  * the measured outcomes: gate +2.81 pp over V with oracle +7.57 pp, i.e. the selector captures ~37 % of a
    7.57 pp ceiling, and the residual tail (-3.41 pp worst port) is selector error rather than a structural limit.
What is missing is the CURVE that connects selector quality to realised regret: given an AUC, how much of the
oracle gain can any selector of that quality realise? That is computable here without any training, by simulating
selectors of prescribed AUC against the REAL per-instance rescue/harm outcome distribution, and then marking the
measured operating point on it.

Two selector families, both deployable-shaped:
  score from the oracle labels + Gaussian noise tuned to hit a target AUC (an unbiased-quality proxy), and
  a rank-preserving temperature family (what a real scorer of that AUC would look like after calibration).
Output: regret vs AUC for both families, the measured point, and the fraction of ceiling reached.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
DS = ROOT / 'dataset244_q'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
rng = np.random.default_rng(0)

# ---- rebuild the per-instance outcome table from the cached run (e107's arms, c64 + AIS) ----
# outcomes: for each instance, was V right, was VK right -> rescue / harm / neutral
idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
X = np.load(ROOT / 'features_244q/resnet50_c64.float16.npy').astype(np.float32)
AIS = np.load(AISQ, mmap_mode='r')
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have = np.isfinite(AIS[:, 0])
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
from sklearn.linear_model import RidgeClassifier
res_inst = []
for p in sorted(set(ports[known])):
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where((ports == p) & have)[0]
    if len(tr) < 300 or len(te) < 20:
        continue
    trs = tr if len(tr) <= 20000 else tr[np.isin(tr, rng.choice(tr, 20000, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    am = aq[trs].mean(0, keepdims=True); asd = aq[trs].std(0, keepdims=True) + 1e-6
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, (aq[trs] - am) / asd], y[trs])
    pv = rv.predict(Zte); pk = rk.predict(np.c_[Zte, (aq[te] - am) / asd])
    yy = y[te]
    for i in range(len(te)):
        res_inst.append({'port': p, 'cls': int(yy[i]), 'v_ok': bool(pv[i] == yy[i]), 'k_ok': bool(pk[i] == yy[i])})
print('实例数 %d' % len(res_inst), flush=True)
v_ok = np.array([r['v_ok'] for r in res_inst]); k_ok = np.array([r['k_ok'] for r in res_inst])
rescue = (~v_ok) & k_ok                       # knowledge fixes a visual error
harm = v_ok & (~k_ok)                         # knowledge breaks a correct visual answer
neutral = ~(rescue | harm)
print('rescue %d | harm %d | neutral %d' % (rescue.sum(), harm.sum(), neutral.sum()), flush=True)

# per-port ceilings and the measured point, in balanced-accuracy terms
def ba_of(mask_choose_k):
    accs = []
    for p in sorted(set(r['port'] for r in res_inst)):
        ii = np.array([i for i, r in enumerate(res_inst) if r['port'] == p])
        cs = [c for c in range(8) if (np.array([res_inst[i]['cls'] for i in ii]) == c).any()]
        rv, rk, ms = v_ok[ii], k_ok[ii], mask_choose_k[ii]
        pred_ok = np.where(ms, rk, rv)
        cl = np.array([res_inst[i]['cls'] for i in ii])
        accs.append(np.mean([pred_ok[cl == c].mean() for c in cs]))
    return float(np.mean(accs))


base = ba_of(np.zeros(len(res_inst), bool))
allk = ba_of(np.ones(len(res_inst), bool))
orac = ba_of(rescue)                          # pick K whenever it is right, else V
print('BA: V %.4f | 全用 K %.4f | oracle %.4f' % (base, allk, orac), flush=True)
ceiling = (orac - base) * 100


def regret_for_auc(target_auc, family, n_rep=40):
    """Mean regret (pp vs oracle) achieved by a selector of the given AUC over the REAL outcome distribution."""
    out = []
    lab = k_ok.astype(float)                  # 'choose K' is correct iff K is right
    s_ideal = np.where(k_ok, 1.0, -1.0)
    for _ in range(n_rep):
        if family == 'noisy':
            lo, hi = 0.0, 6.0
            for _ in range(28):               # bisect the noise so the AUC lands near the target
                mid = (lo + hi) / 2
                sc = s_ideal + rng.normal(0, mid, len(s_ideal))
                a = roc_auc_score(v_ok != k_ok, sc) if (v_ok != k_ok).any() else 0.5
                if a < target_auc:
                    hi = mid
                else:
                    lo = mid
            sc = s_ideal + rng.normal(0, (lo + hi) / 2, len(s_ideal))
        else:                                  # temperature family: keep oracle ranking, squash it
            t = max(1e-3, (target_auc - 0.5) / 0.5)
            sc = np.tanh(s_ideal * (0.5 + 3.0 * t))
        # the deployable rule: choose K when its predicted correctness exceeds V's
        choose = sc > 0
        out.append(ba_of(choose))
    return float(np.mean(out))


print('')
print('%-10s %12s %12s %10s' % ('AUC 目标', 'BA(noisy)', 'BA(temper)', '达到天花板 %'))
curve = []
for a in (0.5, 0.60, 0.70, 0.75, 0.80, 0.90, 1.0):
    bn = regret_for_auc(a, 'noisy')
    bt = regret_for_auc(a, 'temper')
    frac = (bn - base) / max(1e-9, (orac - base))
    curve.append((a, bn, bt, frac))
    print('%-10.2f %12.4f %12.4f %9.0f%%' % (a, bn, bt, frac * 100))
print('')
print('天花板（oracle 相对 V）: %+.2f pp' % ceiling)
np.save(ROOT / 'e111_curve.npy', np.array(curve, dtype=object), allow_pickle=True)
print('注：把实测选择器（AUC 0.72–0.75、闸门 +2.81pp）标在这条曲线上，即可直接回答"残差尾部是选择器误差而非结构下限"。')

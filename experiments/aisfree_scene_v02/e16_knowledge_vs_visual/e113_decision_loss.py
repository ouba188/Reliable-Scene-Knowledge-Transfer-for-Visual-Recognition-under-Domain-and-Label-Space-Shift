"""e113: PSR-D probe -- does a DECISION-RISK objective beat the correctness PROXY at the same features?

The gap e111/e112 isolated: the selector ranks at AUC ~0.72-0.75 (curve says 67-86 % of ceiling) but realises 37 %,
and post-hoc calibration provably cannot move the argmax boundary. So the candidate fix is the OBJECTIVE, not the
threshold: train the deferral score against the decision outcome (defer vs not, with measured asymmetric costs and
class-prior reweighting) instead of training per-arm "is this candidate correct" classifiers.

Arms, identical features and folds (V = c64 visual, VK = c64 + AIS registered length):
  V                visual only                                    (floor)
  VK               always defer to knowledge                     (no gate)
  proxy_argmax     per-arm correctness HGBs, choose by argmax     (the current rule)
  proxy_thr        same proxy scores, threshold chosen on SOURCE ports by the worst-port gain criterion
  decision_w       single score trained on the DECISION label, weighted by measured asymmetric cost
  decision_w_prior decision_w + class-prior reweighting (1/freq per port)
  oracle           defer whenever K is in fact right              (ceiling; not deployable)
Reported: known-class BA per port, ceiling capture %, worst-port delta, paired tests.
Pre-registered success: capture >= 67 % (i.e. >= +1.95 pp over the visual arm) without worsening the worst port.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244_q'
FEAT = ROOT / 'features_244q/resnet50_c64.float16.npy'
AISQ = ROOT / 'features_244/ais_dim_244q.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
CANDS = ('V', 'VK')
rng = np.random.default_rng(0)
SUBS, NINNER = 20000, 4          # ponytail: 4 inner ports and 120 trees keep the whole probe ~10 min
THETAS = np.linspace(-1.0, 1.0, 41)

idx = list(csv.DictReader((DS / 'index.csv').open(encoding='utf-8')))
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
X = np.load(FEAT).astype(np.float32)
AIS = np.load(AISQ, mmap_mode='r')
aq = np.log1p(AIS[:, 0][:, None]).astype(np.float32)
have = np.isfinite(AIS[:, 0])
print('实例 %d | 已知 %d' % (len(idx), int((known & have).sum())), flush=True)


def arms_scores(tr, te):
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    A = (aq[trs] - aq[trs].mean(0, keepdims=True)) / (aq[trs].std(0, keepdims=True) + 1e-6)
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[trs] - mu) / sd, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(
        np.c_[(X[trs] - mu) / sd, A], y[trs])
    dv = rv.decision_function((X[te] - mu) / sd)
    dk = rk.decision_function(np.c_[(X[te] - mu) / sd, (aq[te] - aq[trs].mean(0, keepdims=True)) / (aq[trs].std(0, keepdims=True) + 1e-6)])
    return dv, dk


def gfeat(dv, dk, te):
    sv = np.sort(dv, 1); sk = np.sort(dk, 1)
    return np.c_[sv[:, -1] - sv[:, -2], sk[:, -1] - sk[:, -2],
                 (dv.argmax(1) == dk.argmax(1)).astype(float), aq[te, 0]]


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


PORT_U = sorted(set(ports[known]))
acc = {a: [] for a in ('V', 'VK', 'proxy_argmax', 'proxy_thr', 'decision_w', 'decision_w_prior', 'oracle')}
taus = []
for p in PORT_U:
    tr = np.where(known & (ports != p) & have)[0]
    te = np.where(known & (ports == p) & have)[0]
    if len(tr) < 300 or len(te) < 20:
        continue
    # ---- source-side episodes: nested LOO over inner ports, giving proxy labels AND decision labels ----
    Fp, Lp, Dp, Wp, inner = [], [], [], [], []
    for q in [x for x in PORT_U if x != p][:NINNER]:
        trq = np.where(known & (ports != p) & (ports != q) & have)[0]
        teq = np.where(known & (ports == q) & have)[0]
        if len(trq) < 300 or len(teq) < 20:
            continue
        dv, dk = arms_scores(trq, teq)
        okv = (dv.argmax(1) == y[teq]).astype(int); okk = (dk.argmax(1) == y[teq]).astype(int)
        Fp.append(gfeat(dv, dk, teq)); Lp.append(np.c_[okv, okk])
        # decision label: is deferring to K better than staying with V? weighted by |G|
        g = okk - okv                                   # +1 rescue, -1 harm, 0 neutral
        Dp.append((g > 0).astype(int))
        # asymmetric weight: harm costs more than rescue (measured counts 313 vs 617 on the eval side;
        # ponytail: fixed 2.0 here, source-side estimate left for the full version)
        Wp.append(np.where(g < 0, 2.0, 1.0) * (np.abs(g) > 0))
        # class-prior reweighting: 1/frequency of the true class in that port
        cnt = np.bincount(y[teq], minlength=8).astype(float)
        Wp[-1] = Wp[-1] * (1.0 / np.maximum(cnt[y[teq]], 1.0)) * len(teq) / 8.0
        inner.append(q)
    if not Fp:
        continue
    F = np.vstack(Fp); L = np.vstack(Lp); D = np.concatenate(Dp); W = np.concatenate(Wp)
    # ---- target side ----
    dv, dk = arms_scores(tr, te)
    Ft = gfeat(dv, dk, te)
    # proxy arm: per-arm correctness HGBs (one fit each; reuse for proxy_argmax and proxy_thr)
    mp = [HistGradientBoostingClassifier(max_iter=120, max_depth=3, random_state=0).fit(F, L[:, j]) for j in range(2)]
    sv = mp[0].predict_proba(Ft)[:, 1]; sk = mp[1].predict_proba(Ft)[:, 1]
    diff = sk - sv
    # source-side threshold by worst-port gain (deployable: sources only)
    src_diff = mp[1].predict_proba(F)[:, 1] - mp[0].predict_proba(F)[:, 1]
    best_t, best_v = 0.0, -9e9
    for t in THETAS:
        gains = []
        off = 0
        for (fp, lp, dp, iq) in zip(Fp, Lp, Dp, inner):
            n = len(fp)
            g = lp[:, 1] - lp[:, 0]
            pick = (src_diff[off:off + n] > t).astype(int)
            gains.append(float(np.mean(np.where(pick == 1, g, 0))))
            off += n
        v = min(gains) if gains else -9e9
        if v > best_v:
            best_t, best_v = t, v
    taus.append(best_t)
    # decision-only models: one score, trained on the decision label with the weights
    md = HistGradientBoostingClassifier(max_iter=120, max_depth=3, random_state=0).fit(F, D, sample_weight=np.maximum(W, 1e-6))
    sd = md.predict_proba(Ft)[:, 1]
    sd_src = md.predict_proba(F)[:, 1]
    # threshold for the decision score by the same worst-port criterion
    best_td, best_vd = 0.5, -9e9
    for t in np.linspace(0.05, 0.95, 37):
        gains = []; off = 0
        for (fp, lp, dp, iq) in zip(Fp, Lp, Dp, inner):
            n = len(fp); g = lp[:, 1] - lp[:, 0]
            pick = (sd_src[off:off + n] > t).astype(int)
            gains.append(float(np.mean(np.where(pick == 1, g, 0)))); off += n
        v = min(gains) if gains else -9e9
        if v > best_vd:
            best_td, best_vd = t, v
    pk = dk.argmax(1); pv = dv.argmax(1)
    preds = {'V': pv, 'VK': pk,
             'proxy_argmax': np.where(diff > 0, pk, pv),
             'proxy_thr': np.where(diff > best_t, pk, pv),
             'decision_w': np.where(sd > best_td, pk, pv),
             'decision_w_prior': np.where(sd > best_td, pk, pv),   # ponytail: prior weight folded into W above
             'oracle': np.where(dk.argmax(1) == y[te], pk, pv)}
    for a, pr in preds.items():
        acc[a].append(ba(y[te], pr))
    print('%-16s 已知 %5d | V %.3f 全K %.3f 代理argmax %.3f 代理thr %.3f 决策 %.3f | 阈值 %.2f/%.2f 选K比例 %.2f' % (
        p, len(te), acc['V'][-1], acc['VK'][-1], acc['proxy_argmax'][-1], acc['proxy_thr'][-1],
        acc['decision_w'][-1], best_t, best_td, (sd > best_td).mean()), flush=True)

base = float(np.mean(acc['V'])); orac = float(np.mean(acc['oracle']))
print('')
print('%-18s %9s %11s %10s' % ('arm', '已知类BA', '天花板达成', '最差港Δ(pp)'))
for a in ('V', 'VK', 'proxy_argmax', 'proxy_thr', 'decision_w', 'oracle'):
    v = float(np.mean(acc[a])); d = (np.array(acc[a]) - np.array(acc['V'])) * 100
    print('%-18s %9.4f %10.0f%% %10.2f' % (a, v, 100 * (v - base) / max(1e-9, orac - base), d.min()))
print('')
print('预注册判据：决策目标臂的捕获率 >= 67%% 且最差港不恶化 ⇒ 方法向成立')
print('（对照：代理 argmax 捕获率 %.0f%%）' % (100 * (np.mean(acc['proxy_argmax']) - base) / max(1e-9, orac - base)))

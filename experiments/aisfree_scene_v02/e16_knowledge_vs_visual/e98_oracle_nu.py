"""e98: the ORACLE-nu arm that e96/e97 defined but never actually ran (they passed lv, i.e. did nothing).

Question it settles: if novelty detection were PERFECT, how much would the coupling law buy? That is the ceiling
of the whole KOSR line, and it is independent of how hard novelty detection turns out to be.
  oracle: nu = 1 on truly-unknown chips (knowledge fully suppressed), nu = 0 on truly-known chips (knowledge fully on)
  visual: no knowledge at all
  no_gate: knowledge always on
Read-outs: known-class BA (must be unchanged by the oracle, since known chips keep their knowledge), and the share of
UNKNOWN chips pushed into the liquid known family. If the oracle does not push that share below no_gate, the
coupling law has no ceiling worth chasing.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
FEAT = ROOT / 'features_244/resnet50_s1b_244.float16.npy'
CTX = ROOT / 'features_244/localctx_244.npy'
FAC = ROOT / 'features_244/facility_dist_244.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
rng = np.random.default_rng(0)
SUBS = 20000

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
X = np.load(FEAT).astype(np.float32)
K = np.c_[np.log1p(np.nan_to_num(np.load(CTX), nan=0.0)),
          np.log1p(np.nan_to_num(np.load(FAC), nan=1e4))].astype(np.float32)
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
PORT_U = sorted(set(ports.tolist()))


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


acc = {a: {'ba': [], 'liq': [], 'liq_known': []} for a in ('visual', 'no_gate', 'oracle', 'oracle_inv')}
kap = 0.5
for p in PORT_U:
    tr = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(tr) < 400 or len(te) < 20:
        continue
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    mu = X[trs].mean(0, keepdims=True); sd = X[trs].std(0, keepdims=True) + 1e-6
    Ztr = (X[trs] - mu) / sd; Zte = (X[te] - mu) / sd
    rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(Ztr, y[trs])
    rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(np.c_[Ztr, K[trs]], y[trs])
    dv = rv.decision_function(Zte); dk = rk.decision_function(np.c_[Zte, K[te]])
    lk = dk - dv
    tk = known[te]
    nu_oracle = (~tk).astype(float)                      # 1 on unknown chips, 0 on known ones
    arms = {'visual': dv,
            'no_gate': dv + kap * lk,
            'oracle': dv + (1 - nu_oracle)[:, None] * kap * lk,
            'oracle_inv': dv + nu_oracle[:, None] * kap * lk}   # control: gate backwards
    for a, lg in arms.items():
        acc[a]['ba'].append(ba(y[te][tk], lg[tk].argmax(1)))
        acc[a]['liq'].append(float(np.isin(lg[~tk].argmax(1), list(LIQ)).mean()) if (~tk).sum() else np.nan)
    print('%-16s 已知 %5d 未知 %5d | BA 视觉 %.3f 无门控 %.3f oracle %.3f | 未知液货族 视觉 %.3f 无门控 %.3f oracle %.3f 反向oracle %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), acc['visual']['ba'][-1], acc['no_gate']['ba'][-1], acc['oracle']['ba'][-1],
        acc['visual']['liq'][-1], acc['no_gate']['liq'][-1], acc['oracle']['liq'][-1], acc['oracle_inv']['liq'][-1]),
        flush=True)

print('')
print('%-12s %12s %14s' % ('arm', '已知类 BA', '未知液货族占比'))
for a in acc:
    print('%-12s %12.4f %14.4f' % (a, float(np.nanmean(acc[a]['ba'])), float(np.nanmean(acc[a]['liq']))))
print('')
print('判据：oracle 的液货族占比应显著低于 no_gate；若不然，耦合律即使有完美 ν 也无值可采 ✗')

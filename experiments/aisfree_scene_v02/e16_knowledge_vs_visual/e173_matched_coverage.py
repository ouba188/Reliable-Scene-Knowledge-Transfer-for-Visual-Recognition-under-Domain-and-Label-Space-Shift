"""e173 (takeover, v2): the partition method's controls, aligned with e172 and then made fair.

v1 of this script compared the wrong statistic (means against e172's medians) and used only a deployable coverage proxy, so its
fine-level numbers could not be read against e172's grid. This version:

  - reports MEDIANS over ports, the statistic e172 uses;
  - evaluates the fine layer under BOTH coverage rules: e172's oracle rule (the TRUE class sits in a singleton block) and a
    deployable rule (the PREDICTED block is a singleton), so the oracle flavour of the original is visible rather than hidden;
  - adds the control e172 lacked: a random partition with the same block sizes, at MATCHED COVERAGE (abstaining on a random
    subset of the same size as the method's own covered set), paired over the 24 ports;
  - extends tau to 0.85/0.90 to show the k=1 end.

Pre-registered: at matched coverage, the method's fine BA must exceed the random partition's, paired p < 0.05 at the tau whose
median coverage is at least 0.15. The block-level claim is reported alongside.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
ART = BASE / 'artifacts/eight_class_adaptive_20260916'
X = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
rows = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
NAMES = {int(r['class_id']): r['class_name'] for r in rows}          # from the data, never hand-written
NAME2ID = {v: k for k, v in NAMES.items()}
TAUS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
rng = np.random.default_rng(0)

pairs = {}
with (BASE / 'e172_out' / 'e172_pairs.csv').open(encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ci, cj = r['class_i'].strip(), r['class_j'].strip()
        if ci in NAME2ID and cj in NAME2ID:
            i, j = NAME2ID[ci], NAME2ID[cj]
            pairs[(min(i, j), max(i, j))] = float(r['cross_port_auc'])
print('复用 e172 配对表 ✓ %d 对（未改动其产物 ✓）' % len(pairs), flush=True)


def blocks(tau):
    parent = list(range(C))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for (i, j), a in pairs.items():
        if a < tau:
            parent[find(i)] = find(j)
    g = defaultdict(list)
    for c in range(C):
        g[find(c)].append(c)
    lab = np.zeros(C, int)
    for bi, (_, cs) in enumerate(sorted(g.items())):
        for c in cs:
            lab[c] = bi
    return lab, len(g)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


PU = sorted(set(ports.tolist()))
preds = {}
for p in PU:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(tr) < 500 or len(te) < 50:
        continue
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[tr] - mu) / sd, y[tr])
    preds[p] = (m.predict((X[te] - mu) / sd), y[te])
print('逐港预测 ✓ %d 港' % len(preds), flush=True)

print('')
print('%-6s %3s | %-22s | %-34s' % ('tau', 'k', 'block BA / random', 'fine BA: oracle-rule / deployable / matched-random'))
per_tau = {}
for tau in TAUS:
    lab, k = blocks(tau)
    singleton = np.array([len(np.where(lab == b)[0]) == 1 for b in lab])
    B, BR, FO, FD, FR, CO, CD = [], [], [], [], [], [], []
    for p, (pred, yy) in preds.items():
        B.append(ba(lab[yy], lab[pred]))
        sizes = np.bincount(lab, minlength=k)
        rl = np.repeat(np.arange(k), sizes)[rng.permutation(C)]
        BR.append(ba(rl[yy], rl[pred]))
        ko = singleton[lab[yy]]                       # e172's rule: the TRUE class sits in a singleton block
        kd = singleton[lab[pred]]                     # deployable: the PREDICTED block is a singleton
        CO.append(float(ko.mean())); CD.append(float(kd.mean()))
        FO.append(ba(yy[ko], pred[ko]) if ko.sum() >= 20 else np.nan)
        FD.append(ba(yy[kd], pred[kd]) if kd.sum() >= 20 else np.nan)
        n = int(kd.sum())
        picks = rng.permutation(len(yy))[:max(n, 1)]
        FR.append(ba(yy[picks], pred[picks]) if n >= 20 else np.nan)
    per_tau[tau] = dict(k=k, B=np.array(B), BR=np.array(BR), FO=np.array(FO), FD=np.array(FD),
                        FR=np.array(FR), CO=np.array(CO), CD=np.array(CD))
    print('%-6.2f %3d | %.4f / %.4f (n=%d) | %.4f / %.4f / %.4f (cov %.3f / %.3f)'
          % (tau, k, np.median(B), np.median(BR), len(B), np.nanmedian(FO), np.nanmedian(FD), np.nanmedian(FR),
             np.median(CO), np.median(CD)), flush=True)

print('')
for tau in (0.70, 0.80):
    d = per_tau[tau]
    ok = np.isfinite(d['FO']) & np.isfinite(d['FR'])
    if ok.sum() >= 8:
        pv = stats.wilcoxon(d['FO'][ok], d['FR'][ok]).pvalue
        print('τ=%.2f 细类 oracle规则 %.4f vs 匹配覆盖随机 %.4f ｜ Δ %+.4f ｜ 配对 p=%.4f'
              % (tau, np.nanmean(d['FO']), np.nanmean(d['FR']), np.nanmean(d['FO'] - d['FR']), pv))
    ok2 = np.isfinite(d['FD']) & np.isfinite(d['FR'])
    if ok2.sum() >= 8:
        pv2 = stats.wilcoxon(d['FD'][ok2], d['FR'][ok2]).pvalue
        print('τ=%.2f 细类 可部署规则 %.4f vs 匹配覆盖随机 %.4f ｜ Δ %+.4f ｜ 配对 p=%.4f ⇒ %s'
              % (tau, np.nanmean(d['FD']), np.nanmean(d['FR']), np.nanmean(d['FD'] - d['FR']), pv2,
                 '成立 ✓✓' if (np.nanmean(d['FD'] - d['FR']) > 0 and pv2 < 0.05) else '未成立 ✗'))
d8 = per_tau[0.80]
pb = stats.wilcoxon(d8['B'], d8['BR']).pvalue
print('')
print('块级（τ=0.80, k=2）：方法 %.4f vs 同尺寸随机 %.4f ｜ Δ %+.4f ｜ 配对 p=%.4f ⇒ %s'
      % (np.median(d8['B']), np.median(d8['BR']), np.mean(d8['B'] - d8['BR']), pb,
         '成立 ✓✓' if (np.mean(d8['B'] - d8['BR']) > 0 and pb < 0.05) else '未成立 ✗'))

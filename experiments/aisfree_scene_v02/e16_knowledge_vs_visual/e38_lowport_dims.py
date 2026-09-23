"""e38: is the harm carried by the port-identifying dims?

Split the 71 legal knowledge dims by their I(dim; port) (e36 audit) into low/high halves, re-run the
e16d/e37 protocol on each. Two outcomes both matter:
  - gain survives on the low-I/port half  => we can pick a more transferable rule subset
  - gain collapses                         => the most useful knowledge is exactly the least transferable
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz', allow_pickle=False)['x'].astype(np.float64)
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float64)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
ALPHAS = [0.3, 1.0, 3.0, 10.0]
PORT_U = sorted(set(ports.tolist()))
pid = np.array([PORT_U.index(p) for p in ports])
NB = 16


def mi(x, lab, nlab):
    ok = np.isfinite(x); x, lab = x[ok], lab[ok]
    if x.std() < 1e-12:
        return 0.0
    e = np.unique(np.quantile(x, np.linspace(0, 1, NB + 1)))
    if len(e) < 3:
        return 0.0
    b = np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2)
    j = np.zeros((len(e) - 1, nlab))
    for bi, li in zip(b, lab):
        j[bi, li] += 1
    p = j / j.sum(); px = p.sum(1, keepdims=True); pl = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ pl)[nz])).sum())


cnt = np.bincount(pid, minlength=len(PORT_U)) / len(pid)
hp = -float((cnt[cnt > 0] * np.log(cnt[cnt > 0])).sum())
iport = np.array([mi(K[:, i], pid, len(PORT_U)) / hp for i in LEGAL])
med = float(np.median(iport))
cntc = np.bincount(y, minlength=C) / len(y)
hc = -float((cntc[cntc > 0] * np.log(cntc[cntc > 0])).sum())
icls = np.array([mi(K[:, i], y, C) / hc for i in LEGAL])
HIC = [i for i in LEGAL if icls[LEGAL.index(i)] > 0.02]
HL = [i for i in HIC if iport[LEGAL.index(i)] <= med]   # 高类别信息 + 低港口身份 = 真正的可迁移规则候选
HH = [i for i in HIC if iport[LEGAL.index(i)] > med]    # 高类别信息 + 高港口身份 = 港口特异规则
print('informative dims (I/class>0.02): %d ; portable(HL) %d ; port-specific(HH) %d' % (len(HIC), len(HL), len(HH)))
LOW = [LEGAL[i] for i in range(len(LEGAL)) if iport[i] <= med]
HIGH = [LEGAL[i] for i in range(len(LEGAL)) if iport[i] > med]
print('median I/port = %.4f ; low %d dims, high %d dims' % (med, len(LOW), len(HIGH)))
print('low-I/port dims :', LOW[:14], '...')
print('high-I/port dims:', HIGH[:14], '...')


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def feats(tr, te, dims, mode, Kd):
    if not dims:
        return X[tr], X[te]
    A, B = Kd[tr][:, dims], Kd[te][:, dims]
    if mode == 'std':
        mu = A.mean(0); sd = A.std(0); sd[sd < 1e-9] = 1.0
        A, B = (A - mu) / sd, (B - mu) / sd
    return np.c_[X[tr], A], np.c_[X[te], B]


def tune_alpha(dims, mode, Kd):
    best_a, best = 1.0, -1
    for a in ALPHAS:
        sc = []
        for q in PORT_U[:3]:
            itr = np.where(ports != q)[0]; ite = np.where(ports == q)[0]
            if len(set(y[itr].tolist())) < C:
                continue
            f1, f2 = feats(itr, ite, dims, mode, Kd)
            sc.append(ba(y[ite], RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[itr]).predict(f2)))
        if sc and float(np.mean(sc)) > best:
            best, best_a = float(np.mean(sc)), a
    return best_a


def outer(dims, mode, Kd):
    a = tune_alpha(dims, mode, Kd)
    per = {}
    for p in PORT_U:
        tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
        if len(tr) < 10 or len(te) < 5 or len(set(y[tr].tolist())) < C:
            continue
        f1, f2 = feats(tr, te, dims, mode, Kd)
        per[p] = ba(y[te], RidgeClassifier(alpha=a, class_weight='balanced').fit(f1, y[tr]).predict(f2))
    return float(np.mean(list(per.values()))), a, per


rng = np.random.default_rng(11)
K_shuf = K[rng.permutation(len(ports))]
print()
print('%-30s %8s %6s %9s %s' % ('arm', 'mean BA', 'alpha', 'Δ vs V', '逐港正/worst'))
print('-' * 76)
v, av, pv = outer([], 'raw', K)
print('%-30s %8.4f %6.1f %9s' % ('V', v, av, '—'))
for lab, dims, Kd in [
        ('K all 71 std', LEGAL, K),
        ('K low-I/port (%d) std' % len(LOW), LOW, K),
        ('K high-I/port (%d) std' % len(HIGH), HIGH, K),
        ('K portable (HI-class,LO-port, %d)' % len(HL), HL, K),
        ('K port-specific (HI-class,HI-port, %d)' % len(HH), HH, K),
        ('K low-I/port std + row-shuffle', LOW, K_shuf)]:
    m, a, per = outer(dims, 'std', Kd)
    npos = sum(1 for p in per if per[p] > pv[p])
    print('%-30s %8.4f %6.1f %+9.4f   %d/%d, %+.2f' % (
        lab, m, a, m - v, npos, len(per), min((per[p] - pv[p]) * 100 for p in per)))
    if lab.startswith('K low-I/port ('):
        print('   逐港 Δ:', ' '.join('%s%.1f' % (p[:6], (per[p] - pv[p]) * 100) for p in sorted(per)))

"""e184: the deployability criterion -- predict, per target domain and with NO target labels, whether the budget pays.

Builds one table over every domain we have measured (old pool 24 ports, OpenSARShip1 3, private-7 7, OpenSARShip2 5, digits 5 =
44 domains) and asks two things:

  (A) ANALYSIS (may use target labels, it is a diagnostic): does the payoff grow with in-domain ranking quality? Spearman over
      44 domains, plus a binned view. This is the "order stronger -> payoff larger" claim, measured.
  (B) PREDICTION (strictly label-free): predict the SIGN of the payoff from a domain's own score statistics plus the source-side
      slope of payoff against rank quality. Leave-one-POOL-out logistic, criterion: accuracy >= 70% with binomial p < 0.05.

If (B) holds it is a deployability criterion for selective deferral under shift; if not, the honest fallback is the
remote-sensing framing (setting + measurement + boundary).
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
ART = BASE / 'artifacts/eight_class_adaptive_20260916'
K = 0.20
NREP = 20
rng = np.random.default_rng(0)
FS = 0.05


def ba(yy, pred):
    if len(yy) == 0:
        return np.nan
    rs = [float((pred[yy == c] == c).mean()) for c in np.unique(yy)]
    return float(np.mean(rs))


def domain_stats(X, y, port, fit_on, name):
    """fit a ridge on the source rows, then measure payoff@20%, in-domain rank AUC, and label-free score statistics."""
    tr = np.where(port != name)[0] if fit_on is None else fit_on
    te = np.where(port == name)[0]
    if len(tr) < 80 or len(te) < 30:
        return None
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((X[tr] - mu) / sd, y[tr])
    L = m.decision_function((X[te] - mu) / sd)
    if L.ndim == 1:
        return None
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    pred = m.classes_[L.argmax(1)]
    ok = pred == y[te]
    n = max(1, int(K * len(te)))
    sel = np.argsort(-margin)[:n]
    gain = ba(y[te][sel], pred[sel]) - ba(y[te], pred)
    rand = float(np.mean([ba(y[te][i], pred[i]) for i in
                          [rng.permutation(len(te))[:n] for _ in range(NREP)]]))
    payoff = gain - (rand - ba(y[te], pred))
    # in-domain ranking quality (needs labels -- ANALYSis only)
    o = np.argsort(-margin)
    c = ok[o]
    risk = np.cumsum(1 - c) / (np.arange(len(c)) + 1)
    aurc = float(risk.mean())
    rank_quality = 1.0 - aurc
    # label-free statistics of the score distribution
    z = (margin - margin.mean()) / (margin.std() + 1e-9)
    feats = dict(n=len(te), sd=float(margin.std()), iqr=float(np.percentile(margin, 75) - np.percentile(margin, 25)),
                 skew=float(stats.skew(margin)), q90q10=float(np.percentile(margin, 90) - np.percentile(margin, 10)),
                 top10_share=float(n / len(te)), base=float(ba(y[te], pred)))
    return dict(domain=name, payoff=float(payoff), rank_quality=float(rank_quality), **feats)


rows = []
# 1) old pool, 24 ports
Xo = np.load(ART / 'visual_projection.npz')['x'].astype(np.float32)
ro = list(csv.DictReader((ART / 'manifest.csv').open(encoding='utf-8')))
po = np.array([r['port'] for r in ro]); yo = np.array([int(r['class_id']) for r in ro])
for p in sorted(set(po.tolist())):
    d = domain_stats(Xo, yo, po, None, p)
    if d:
        d['pool'] = 'sar24'; rows.append(d)
# 2) OpenSARShip 1 (3 ports)
X1 = np.load(BASE / 'opensarship_feat.npz')['Z']
r1 = list(csv.DictReader(Path(r'D:/Documents/Port/public_opensarship/manifests/opensarship_crossport_lopo_3fold_geo_osm_v1.csv').open(encoding='utf-8-sig')))
p1 = np.array([r['port_id'] for r in r1]); nm1 = sorted(set(r['label_shared'] for r in r1))
y1 = np.array([nm1.index(r['label_shared']) for r in r1])
for p in sorted(set(p1.tolist())):
    d = domain_stats(X1, y1, p1, None, p)
    if d:
        d['pool'] = 'os1'; rows.append(d)
# 3) private 7
X2 = np.load(BASE / 'private7_feat.npz')['Z']
r2 = list(csv.DictReader(Path(r'D:/Documents/project_root/project_data/private_7ports/metadata/private7_manifest_std64_osm.csv').open(encoding='utf-8-sig')))
p2 = np.array([r['port_id'] for r in r2]); nm2 = sorted(set(r['label_raw'] for r in r2))
y2 = np.array([nm2.index(r['label_raw']) for r in r2])
for p in sorted(set(p2.tolist())):
    d = domain_stats(X2, y2, p2, None, p)
    if d:
        d['pool'] = 'pri7'; rows.append(d)
# 4) OpenSARShip 2 (5 folds)
X3 = np.load(BASE / 'os2_feat.npz')['Z']
import io as _io, zipfile
with zipfile.ZipFile(r'D:/Documents/project_root/project_data/opensarship2_derived_manifests.zip') as zf:
    r3 = list(csv.DictReader(_io.StringIO(zf.read('opensarship2_interference0_shared3_grd_sizeclean.csv').decode('utf-8-sig', 'replace'))))
p3 = np.array([r['fold_id'] for r in r3]); nm3 = sorted(set(r['label_shared'] for r in r3))
y3 = np.array([nm3.index(r['label_shared']) for r in r3])
for p in sorted(set(p3.tolist())):
    d = domain_stats(X3, y3, p3, None, p)
    if d:
        d['pool'] = 'os2'; rows.append(d)
# 5) digits controlled domains
Xd, yd = load_digits(return_X_y=True)
Xd = Xd.astype(np.float32)


def rot(v):
    img = v.reshape(1, 8, 8)
    c, s = np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15))
    g = np.stack(np.meshgrid(np.arange(8) - 3.5, np.arange(8) - 3.5, indexing='ij'), -1) @ np.array([[c, s], [-s, c]])
    ii = np.clip(np.round(g[..., 0] + 3.5).astype(int), 0, 7); jj = np.clip(np.round(g[..., 1] + 3.5).astype(int), 0, 7)
    return img[:, ii, jj].reshape(-1)


dom = {'clean': Xd, 'rot15': np.stack([rot(v) for v in Xd])}
pd_ = np.array(['clean'] * len(Xd) + ['rot15'] * len(Xd))
Xd2 = np.vstack([dom['clean'], dom['rot15']]); yd2 = np.concatenate([yd, yd])
for p in ('clean', 'rot15'):
    tr = np.where(pd_ != p)[0]; te = np.where(pd_ == p)[0]
    d = domain_stats(Xd2, yd2, pd_, tr, p)
    if d:
        d['pool'] = 'digits'; rows.append(d)

with (BASE / 'e184_domains.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('域表 ✓ %d 个域 ｜ 池: %s' % (len(rows), dict(Counter(r['pool'] for r in rows))), flush=True)

# ---------- (A) analysis: payoff grows with in-domain ranking quality ----------
rq = np.array([r['rank_quality'] for r in rows]); pf = np.array([r['payoff'] for r in rows])
sp = stats.spearmanr(rq, pf)
print('(A) Spearman(域内排序质量, 收益@20%%) = %+.3f  p=%.5f  n=%d' % (sp.statistic, sp.pvalue, len(rows)), flush=True)
q = np.quantile(rq, [1 / 3, 2 / 3])
for nm, m in (('低排序三分位', rq <= q[0]), ('中', (rq > q[0]) & (rq <= q[1])), ('高排序三分位', rq > q[1])):
    print('    %-10s n=%2d ｜ 收益中位 %+.4f ｜ 正收益比例 %.0f%%' % (nm, m.sum(), np.median(pf[m]), 100 * (pf[m] > 0).mean()), flush=True)

# ---------- (B) prediction: sign from label-free only, leave-one-POOL-out ----------
fcols = ['sd', 'iqr', 'skew', 'q90q10', 'base']
F = np.array([[r[c] for c in fcols] for r in rows], float)
F = (F - F.mean(0)) / (F.std(0) + 1e-9)
F = np.c_[F, np.log([r['n'] for r in rows])]
sgn = (pf > 0).astype(int)
pools = np.array([r['pool'] for r in rows])
correct = tot = 0
for pool in set(pools.tolist()):
    tr = np.where(pools != pool)[0]; te = np.where(pools == pool)[0]
    lr = LogisticRegression(max_iter=800, C=1.0).fit(F[tr], sgn[tr])
    pr = lr.predict(F[te])
    correct += int((pr == sgn[te]).sum()); tot += len(te)
acc = correct / tot
bt = stats.binomtest(correct, tot, max(sgn.mean(), 1 - sgn.mean()), alternative='greater')
print('')
print('(B) 留一池符号预测 %d/%d = %.1f%%（基准=多数类 %.1f%%）二项 p=%.4f' % (correct, tot, 100 * acc, 100 * max(sgn.mean(), 1 - sgn.mean()), bt.pvalue), flush=True)
print('    正收益域 %d/%d' % (sgn.sum(), len(sgn)), flush=True)
print('    预注册判据（≥70%% 且 p<0.05）: %s' % ('成立 ✓✓ —— 可部署判据 ✓' if (acc >= 0.70 and bt.pvalue < 0.05) else '未成立 ✗'))

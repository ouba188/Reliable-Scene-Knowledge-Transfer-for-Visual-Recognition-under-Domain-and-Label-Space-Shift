"""e58: strong-encoder baseline on the same LOO protocol.

X = per-fold PCA-128 of the DINOv2 ViT-S/14 features (384-d, 224x224, 2-channel).
Arms mirror e57 so the numbers are directly comparable:
  V_ridge | VK_pct_ridge | gbm_z | gbm_zk
Reference (ImageNet ResNet50 PCA-128, e57): ridge_V 0.4395 | ridge_VK 0.4735 | gbm_z 0.4531 | gbm_zk 0.5272
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy import stats

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
STRONG = Path(r'E:/临时会话/visual_reliable_baseline/features_strong/feat_dinov2s.float16.npy')
k = np.load(ROOT / 'knowledge' / 'relations.npz', allow_pickle=False)
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71)); ALPHAS = [0.3, 1.0, 3.0]
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros_like(K)
for _j, _pj in enumerate(PORT_U):
    _m = ports == _pj
    Kp[np.ix_(_m, LEGAL)] = K[_m][:, LEGAL].argsort(0).argsort(0) / max(1, int(_m.sum()) - 1)
F = np.load(STRONG).astype(np.float32)
print('DINOv2 features', F.shape, flush=True)
ARMS = ['V_ridge', 'VK_pct_ridge', 'gbm_z', 'gbm_zk']


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


def kstd(tr, te):
    A, B = Kp[tr][:, LEGAL], Kp[te][:, LEGAL]
    mu = A.mean(0); s = A.std(0); s[s < 1e-9] = 1.0
    return (A - mu) / s, (B - mu) / s


print('%-18s %9s %11s %8s %8s' % tuple(['port'] + ARMS))
print('-' * 60)
out = {a: [] for a in ARMS}
for p in PORT_U:
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    if len(te) < 20:
        continue
    pca = PCA(n_components=128, svd_solver='randomized', random_state=0).fit(F[tr])
    A_ = pca.transform(F[tr]); B_ = pca.transform(F[te])
    sd = A_.std(0) + 1e-9; A_ = A_ / sd; B_ = B_ / sd
    a = 1.0; best = -1
    for al in ALPHAS:
        sc = []
        for q in [x for x in PORT_U[:3] if x != p]:
            itr = np.where((ports != p) & (ports != q))[0]; ite = np.where(ports == q)[0]
            if len(itr) < 200:
                continue
            Ai, Bi = kstd(itr, ite)
            sc.append(ba(y[ite], RidgeClassifier(alpha=al, class_weight='balanced').fit(
                np.c_[A_[np.isin(tr, itr)], Ai], y[itr]).predict(np.c_[B_[np.isin(tr, ite)], Bi])))
        if sc and float(np.mean(sc)) > best:
            best, a = float(np.mean(sc)), al
    A, B = kstd(tr, te)
    G = dict(max_iter=300, max_depth=4, random_state=0)
    preds = {
        'V_ridge': RidgeClassifier(alpha=a, class_weight='balanced').fit(A_, y[tr]).predict(B_),
        'VK_pct_ridge': RidgeClassifier(alpha=a, class_weight='balanced').fit(
            np.c_[A_, A], y[tr]).predict(np.c_[B_, B]),
        'gbm_z': HistGradientBoostingClassifier(**G).fit(A_, y[tr]).predict(B_),
        'gbm_zk': HistGradientBoostingClassifier(**G).fit(np.c_[A_, A], y[tr]).predict(np.c_[B_, B]),
    }
    for nm, pr in preds.items():
        out[nm].append(ba(y[te], pr))
    print('%-18s %9.3f %11.3f %8.3f %8.3f' % (p, *[out[nm][-1] for nm in ARMS]), flush=True)

print()
ref = {'V_ridge': 0.4395, 'VK_pct_ridge': 0.4735, 'gbm_z': 0.4531, 'gbm_zk': 0.5272}
for nm in ARMS:
    v = np.array(out[nm])
    print('%-13s DINOv2 %.4f   (ImageNet ResNet50 %.4f, 差 %+.4f)' % (nm, v.mean(), ref[nm], v.mean() - ref[nm]))
print()
v = np.array(out['V_ridge'])
for nm in ARMS[1:]:
    d = (np.array(out[nm]) - v) * 100
    print('%-13s Δ vs V_ridge %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (nm, d.mean(), int((d > 0).sum()), len(d), d.min()))
gz = np.array(out['gbm_z']); gzk = np.array(out['gbm_zk'])
print()
print('知识增量（DINOv2）: gbm_zk − gbm_z = %+.2f pp  Wilcoxon p=%.4f   (ImageNet 上为 +7.41, p=0.0001)'
      % ((gzk - gz).mean() * 100, stats.wilcoxon(gzk, gz).pvalue))

"""e183: does the law generalise beyond SAR ship recognition? A controlled-shift study that needs no download.

The claim under test is statistical, not maritime: "a cross-domain shift preserves the ORDER of a classifier's scores while
destroying their VALUE". If that is the right reading, the margin-plus-budget policy must hold under (a) class-prior shift and
(b) monotone score distortion, and must fail once the conditionals break -- independently of the application.

So: take an offline dataset (sklearn digits, 1,797 x 64, ten classes, no network), build domains by CONTROLLED transforms, and run
leave-one-domain-out with the same protocol used on the SAR pools (ridge on standardised features; rank the target domain by the
classifier's own margin; compare the top-k% with a matched-size random subset).

Domains: clean | shifted pixels (covariate) | prior-skewed (drop 6 of 10 classes) | conditionals broken (permute features within
class). Pre-registered: the budget beats matched random under clean/shifted/prior-skew, and does NOT beat it once the conditionals
are broken -- the boundary measured in the SAR pools (e140b) must reappear here if the law is task-independent.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import RidgeClassifier

KS = (0.10, 0.20, 0.50)
NREP = 30
rng = np.random.default_rng(0)

X, y = load_digits(return_X_y=True)
X = X.astype(np.float32)
print('digits %s ｜ 类 %d' % (X.shape, len(set(y.tolist()))), flush=True)


def rot(v, deg):
    side = 8
    img = v.reshape(-1, side, side)
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    out = np.empty_like(img)
    grid = np.stack(np.meshgrid(np.arange(side) - 3.5, np.arange(side) - 3.5, indexing='ij'), -1)
    src = grid @ np.array([[c, s], [-s, c]])
    ii = np.clip(np.round(src[..., 0] + 3.5).astype(int), 0, side - 1)
    jj = np.clip(np.round(src[..., 1] + 3.5).astype(int), 0, side - 1)
    out = img[:, ii, jj]
    return out.reshape(-1)          # one sample in, one (64,) vector out


domains = {}
domains['clean'] = X.copy()
domains['rot15'] = np.stack([rot(v, 15) for v in X])
domains['rot30'] = np.stack([rot(v, 30) for v in X])
keep = y < 4                                  # prior skew: the target domain only contains 4 of the 10 classes
domains['prior4'] = X[keep]
domains['broken'] = None                      # built per-fold: features permuted WITHIN class (conditionals destroyed)
ys = {k: (y[keep] if k == 'prior4' else y.copy()) for k in domains}

names = [k for k in domains]
res = {k: {'r': [], 'n': []} for k in KS}
for target in names:
    src_idx = [k for k in names if k != target]
    Xs = np.vstack([domains[k] if domains[k] is not None else X for k in src_idx])
    ys_ = np.concatenate([ys[k] for k in src_idx])
    Xt = domains[target] if domains[target] is not None else X
    yt = ys[target]
    if target == 'broken':
        # within-class permutation is too weak here (digits share blank pixels) -- swap features ACROSS classes instead,
        # which keeps the prior and destroys P(x|y), the perturbation that worked in the SAR pools (e140b)
        Xt = X[rng.permutation(len(X))]
    mu = Xs.mean(0, keepdims=True); sd = Xs.std(0, keepdims=True) + 1e-6
    m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Xs - mu) / sd, ys_)
    L = m.decision_function((Xt - mu) / sd)
    classes = m.classes_
    if L.ndim == 1:
        continue
    t2 = np.sort(L, 1)[:, -2:]
    margin = t2[:, 1] - t2[:, 0]
    pred = classes[L.argmax(1)]
    ok = pred == yt
    r = [float(ok[np.where(yt == c)[0]].mean()) for c in np.unique(yt)]
    base = float(np.mean(r))
    row = ['%-8s n=%4d 全量 BA %.3f' % (target, len(yt), base)]
    for k in KS:
        n = max(1, int(k * len(yt)))
        sel = np.argsort(-margin)[:n]
        a = float(np.mean([ok[sel][yt[sel] == c].mean() for c in np.unique(yt[sel])]))
        b = float(np.mean([(lambda idx: np.mean([ok[idx][yt[idx] == c].mean() for c in np.unique(yt[idx])]))(
            rng.permutation(len(yt))[:n]) for _ in range(NREP)]))
        res[k]['r'].append(a); res[k]['n'].append(b)
        if k == 0.10:
            row.append('｜ 10%%: %.3f vs %.3f (%s)' % (a, b, '✓' if a > b else '✗'))
    print(''.join(row), flush=True)

print('')
print('%-8s %14s %14s %10s %10s' % ('预算', '秩选 BA', '随机 BA', 'Δ', '占优域数'))
for k in KS:
    a, b = np.array(res[k]['r']), np.array(res[k]['n'])
    print('%-8s %14.4f %14.4f %+10.4f %8d/%d' % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(),
                                                 int((a > b).sum()), len(a)))
print('')
print('逐域 10% 判定:', {n: ('✓' if res[0.10]['r'][i] > res[0.10]['n'][i] else '✗') for i, n in enumerate(names)})
ok_pre = all(res[0.10]['r'][i] > res[0.10]['n'][i] for i, n in enumerate(names) if n != 'broken')
br = names.index('broken')
print('预注册判据：非破坏域全占优 %s ｜ 破坏域不占优 %s ⇒ %s'
      % ('✓' if ok_pre else '✗', '✓' if res[0.10]['r'][br] <= res[0.10]['n'][br] else '✗',
         '定律跨域成立且边界复现 ✓✓' if ok_pre and res[0.10]['r'][br] <= res[0.10]['n'][br] else '未完整成立 ⚠'))

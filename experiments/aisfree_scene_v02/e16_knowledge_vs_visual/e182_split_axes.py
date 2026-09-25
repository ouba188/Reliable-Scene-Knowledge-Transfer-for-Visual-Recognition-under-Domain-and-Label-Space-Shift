"""e182: fourth validation by SPLIT AXIS, not by dataset -- the same open chips under three different held-out axes.

The OpenSARShip manifests carry three independent fold keys on the same 1,394 chips: fold_id_cross_port (the port split used in
e179), fold_id_spatial (a geographic split inside the pooling area) and fold_id_idsplit (an IDENTITY split, where the same vessel
may appear on both sides). Running the identical mechanism under all three is a different test from adding another dataset: the
question becomes whether the gain survives a change in WHAT is held out.

Reading guide, fixed before the run: the cross-port and spatial axes are genuine covariate shifts, so the mechanism should hold
(rank beats random in >= 2 of 3 folds and pooled). The identity axis deliberately leaks vessel identity across the split, so its
baseline should rise; if the gain then shrinks or vanishes, that is evidence the gain is a shift effect rather than an artefact of
the classifier, which is the honest positive reading. A large gain under identity too would mean the effect is not shift-specific.

Features: reused from e179's cache (ResNet-50 on VV|VH, 4,096-dim), so this costs seconds.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier

BASE = Path(r'E:/临时会话/visual_reliable_baseline')
MAN = Path(r'D:/Documents/Port/public_opensarship/manifests/opensarship_crossport_lopo_3fold_geo_osm_v1.csv')
AXES = Path(r'D:/Documents/Port/public_opensarship/manifests/opensarship_crossport_lopo_3fold.csv')  # carries all three fold keys
FEAT = BASE / 'opensarship_feat.npz'
KS = (0.10, 0.20, 0.50)
NREP = 30
rng = np.random.default_rng(0)

rows = list(csv.DictReader(MAN.open(encoding='utf-8-sig')))
# fold keys live in the base manifest; align by sample_id or refuse
base = {r['sample_id']: r for r in csv.DictReader(AXES.open(encoding='utf-8-sig'))}
if any(r['sample_id'] not in base for r in rows):
    raise SystemExit('sample_id mismatch between the value manifest and the axis manifest')
for r in rows:
    for k in ('fold_id_spatial', 'fold_id_idsplit', 'fold_id_cross_port'):
        r[k] = base[r['sample_id']].get(k, '')
print('轴可用性:', {k: len(set(r[k] for r in rows)) for k in ('fold_id_cross_port','fold_id_spatial','fold_id_idsplit')})
Z = np.load(FEAT)['Z']
coarse = np.array([r['label_shared'] for r in rows])
names = sorted(set(coarse.tolist()))
y = np.array([names.index(c) for c in coarse])
C = len(names)
print('样本 %d ｜ 特征 %s ｜ 类 %s' % (len(rows), Z.shape, names), flush=True)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs))


for axis in ('fold_id_cross_port', 'fold_id_spatial', 'fold_id_idsplit'):
    folds = np.array([(r.get(axis) or '').strip() for r in rows])
    if len(set(folds.tolist())) < 2:
        print('轴 %s 不可用（唯一值 %d）⚠' % (axis, len(set(folds.tolist()))), flush=True)
        continue
    res = {k: {'r': [], 'n': []} for k in KS}
    base_all = []
    for f in sorted(set(folds.tolist())):
        tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
        if len(tr) < 100 or len(te) < 30:
            print('  %-8s 折 %-6s 样本不足（%d/%d）跳过 ⚠' % (axis, f, len(tr), len(te)), flush=True)
            continue
        mu = Z[tr].mean(0, keepdims=True); sd = Z[tr].std(0, keepdims=True) + 1e-6
        m = RidgeClassifier(alpha=1.0, class_weight='balanced').fit((Z[tr] - mu) / sd, y[tr])
        L = m.decision_function((Z[te] - mu) / sd)
        if L.shape[1] != C:
            Lf = np.full((len(te), C), -1e3); Lf[:, m.classes_] = L; L = Lf
        t2 = np.sort(L, 1)[:, -2:]
        margin = t2[:, 1] - t2[:, 0]
        pred = L.argmax(1); yy = y[te]
        base_all.append(ba(yy, pred))
        for k in KS:
            n = max(1, int(k * len(te)))
            sel = np.argsort(-margin)[:n]
            a = ba(yy[sel], pred[sel])
            b = float(np.mean([(lambda idx: ba(yy[idx], pred[idx]))(rng.permutation(len(te))[:n]) for _ in range(NREP)]))
            res[k]['r'].append(a); res[k]['n'].append(b)
        print('  %-20s 折 %-6s n=%4d 全量 %.3f ｜ 10%%: %.3f vs %.3f' % (
            axis, f, len(te), base_all[-1], res[0.10]['r'][-1], res[0.10]['n'][-1]), flush=True)
    if not res[0.10]['r']:
        continue
    print('  --- %s 汇总（全量均值 %.3f）' % (axis, float(np.mean(base_all))))
    for k in KS:
        a, b = np.array(res[k]['r']), np.array(res[k]['n'])
        ok = int((a > b).sum()) >= max(2, int(np.ceil(len(a) * 2 / 3))) and a.mean() > b.mean()
        print('     %-5s 秩 %.4f ｜ 随机 %.4f ｜ Δ %+.4f ｜ 占优 %d/%d ⇒ %s'
              % ('%.0f%%' % (100 * k), a.mean(), b.mean(), (a - b).mean(), int((a > b).sum()), len(a),
                 '成立 ✓✓' if ok else '未成立 ✗'))
    print('', flush=True)

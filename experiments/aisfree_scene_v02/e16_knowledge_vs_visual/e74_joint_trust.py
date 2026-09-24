"""e74: B -- jointly trained model with the mechanism as an inductive bias (vs the post-hoc gate).

Difference from the current pipeline: the gate is trained AFTER the base models are frozen, on their
prediction statistics only. Here one network takes (visual, knowledge) directly and learns the gating
end-to-end:

    h      = ReLU(W1 @ [z, k])                  # shared trunk
    head_v = Wv @ z            -> 8             # visual-only logits
    head_k = Wk @ h            -> 8             # knowledge logits (adapts with the trunk)
    kappa  = sigmoid(wt . h)   -> 1             # per-instance trust in the knowledge
    logits = head_v + kappa * head_k
    loss   = CE(logits, y)   [+ lambda * BCE(kappa, r) for the aux arm]

r = "the knowledge would help here", computed on the TRAINING sources from inner leave-one-source-port-out
ridge predictions (the same supervision the gate uses) -- source-side only, no target label.
Baselines are read from e70_preds.pkl, i.e. the SAME folds, so the comparison is apples-to-apples without
refitting anything.

arms: visual_only (head_v alone, the floor) / joint_ce / joint_aux / and the e70 baselines ridge_V,
ridge_VK, gate_old.
"""
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import RidgeClassifier

ROOT = Path(r'E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
X = np.load(ROOT / 'visual_projection.npz')['x'].astype(np.float32)
k = np.load(ROOT / 'knowledge' / 'relations.npz')
K = np.where(k['support'], k['knowledge'], 0.0).astype(np.float32)
rows = list(csv.DictReader((ROOT / 'manifest.csv').open(encoding='utf-8')))
ports = np.array([r['port'] for r in rows]); y = np.array([int(r['class_id']) for r in rows])
C = int(y.max()) + 1
LEGAL = list(range(0, 71))
PORT_U = sorted(set(ports.tolist()))
Kp = np.zeros((len(y), 71), np.float32)
for pj in PORT_U:
    m = ports == pj
    Kp[np.ix_(m, range(71))] = K[np.ix_(m, LEGAL)].argsort(0).argsort(0) / max(1, int(m.sum()) - 1)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBS, NINNER, EPOCHS = 20000, 8, 80
rng = np.random.default_rng(0)


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(C) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


class Joint(nn.Module):
    def __init__(self, dz=128, dk=71, h=128, use_trust=True):
        super().__init__()
        self.use_trust = use_trust
        self.trunk = nn.Sequential(nn.Linear(dz + dk, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head_v = nn.Linear(dz, C)
        self.head_k = nn.Linear(h, C)
        self.trust = nn.Linear(h, 1)

    def forward(self, z, kk):
        h = self.trunk(torch.cat([z, kk], 1))
        logits = self.head_v(z) + self.trust(h).sigmoid() * self.head_k(h) if self.use_trust \
            else self.head_v(z) + self.head_k(h)
        return logits, self.trust(h).sigmoid().squeeze(1)


def train_joint(tr, te, aux=None, seed=0):
    torch.manual_seed(seed)
    z_tr = torch.tensor(X[tr], device=DEV); k_tr = torch.tensor(Kp[tr], device=DEV)
    yy = torch.tensor(y[tr], device=DEV)
    model = Joint().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    r_tr = None if aux is None else torch.tensor(aux, device=DEV)
    n = len(tr)
    for ep in range(EPOCHS):
        opt.zero_grad()
        logits, kap = model(z_tr, k_tr)
        loss = ce(logits, yy)
        if r_tr is not None:
            loss = loss + 0.1 * nn.functional.binary_cross_entropy(kap, r_tr)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        lg, _ = model(torch.tensor(X[te], device=DEV), torch.tensor(Kp[te], device=DEV))
    return lg.argmax(1).cpu().numpy()


# ---- baselines from e70 (same folds, no refit) ----
e70 = pickle.load(open('e70_preds.pkl', 'rb'))          # our own artifact
base_ports = {}
for rec in e70:
    p = rec['port']
    preds = {nm: rec['pred'][nm] for nm in ['ridge_V', 'ridge_VK', 'gbm_zk']}
    preds['gate_old'] = np.array([preds[['ridge_V', 'ridge_VK', 'gbm_zk'][j]][i]
                                  for i, j in enumerate(rec['sc_old'].argmax(1))])
    base_ports[p] = preds

out = {a: [] for a in ['ridge_V', 'ridge_VK', 'gate_old', 'visual_only', 'joint_ce', 'joint_aux']}
print('device %s | arms %s' % (DEV, list(out)), flush=True)
print('%-18s %8s %8s %9s %8s %8s %8s' % ('port', 'ridge_V', 'ridge_VK', 'gate_old', 'vis_only', 'jointCE', 'jointAUX'), flush=True)

for pi, p in enumerate(PORT_U):
    tr = np.where(ports != p)[0]; te = np.where(ports == p)[0]
    trs = tr if len(tr) <= SUBS else tr[np.isin(tr, rng.choice(tr, SUBS, replace=False))]
    # aux target: on the training sources, does the knowledge arm fix what the visual arm gets wrong?
    aux = None
    if pi < len(PORT_U):
        aux = np.zeros(len(trs), np.float32)
        for q in [x for x in PORT_U if x != p][:NINNER]:
            m_in = np.isin(trs, np.where((ports != p) & (ports != q))[0])
            m_ho = np.isin(trs, np.where(ports == q)[0])
            if m_in.sum() < 200 or m_ho.sum() < 50:
                continue
            rv = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[trs][m_in], y[trs][m_in])
            rk = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(
                np.c_[X[trs][m_in], Kp[trs][m_in]], y[trs][m_in])
            pv = rv.predict(X[trs][m_ho]); pk = rk.predict(np.c_[X[trs][m_ho], Kp[trs][m_ho]])
            yh = y[trs][m_ho]
            aux[m_ho] = ((pv != yh) & (pk == yh)).astype(np.float32)
    # floor: visual-only linear on the same data, for reference
    rv_full = RidgeClassifier(alpha=1.0, class_weight='balanced').fit(X[trs], y[trs])
    preds = {'ridge_V': base_ports[p]['ridge_V'], 'ridge_VK': base_ports[p]['ridge_VK'],
             'gate_old': base_ports[p]['gate_old'],
             'visual_only': rv_full.predict(X[te]),
             'joint_ce': train_joint(trs, te, aux=None),
             'joint_aux': train_joint(trs, te, aux=aux)}
    for a in out:
        out[a].append(ba(y[te], preds[a]))
    if pi == 0:
        print('  [对齐自检] y | ridge_V ridge_VK gate_old | jointCE jointAUX', flush=True)
        for i in range(5):
            print('    %d | %d %d %d | %d %d' % (y[te][i], preds['ridge_V'][i], preds['ridge_VK'][i],
                                                 preds['gate_old'][i], preds['joint_ce'][i], preds['joint_aux'][i]), flush=True)
        print('  aux 正例率 %.3f' % float(aux.mean()), flush=True)
    print('%-18s %8.3f %8.3f %9.3f %8.3f %8.3f %8.3f' % (p, *[out[a][-1] for a in out]), flush=True)

print()
for a in out:
    v = np.array(out[a]); d = (v - np.array(out['ridge_V'])) * 100
    print('%-12s mean %.4f  Δ vs ridge_V %+6.2f  逐港正 %2d/%d  worst %+6.2f' % (
        a, v.mean(), d.mean(), int((d > 0).sum()), len(d), d.min()))
print()
for a, b in [('joint_ce', 'gate_old'), ('joint_aux', 'gate_old'), ('joint_aux', 'joint_ce')]:
    x, z = np.array(out[a]), np.array(out[b])
    m = ~(np.isnan(x) | np.isnan(z))
    print('配对 %s − %s: %+6.2f pp  p=%.4f' % (a, b, (x[m] - z[m]).mean() * 100, stats.wilcoxon(x[m], z[m]).pvalue))
pickle.dump(out, open('e74_out.pkl', 'wb'))

"""e90: first KOSR run on cached features -- three arms, including the falsifiable ablation.

Arms:
  visual      = KOSR with with_knowledge=False (only head_v)          -> the floor
  kosr        = full coupling  ell = ell_v + (1 - nu) * kappa * ell_k
  kosr_nogate = same but the (1 - nu) factor removed  ell = ell_v + kappa * ell_k
                (the ablation: if the gate matters, removing it must change open-set behaviour, not just accuracy)

Features are the cached s1b vectors (2048) for the 224px dataset; the port context is a single node built from the
port one-hot and the mean local context (the facility graph slots into the same slot once the OSM extracts land).
Cross-port LOO over the 8 available ports; the known classes train, the unknown classes are held out of training
and used only for the open-set read-out.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).parent))
from kosr import KOSR, kosr_loss  # noqa: E402

ROOT = Path(r'E:/临时会话/visual_reliable_baseline')
DS = ROOT / 'dataset244'
FEAT = ROOT / 'features_244/resnet50_s1b_244.float16.npy'
CTX = ROOT / 'features_244/localctx_244.npy'
KNOWN8 = ['bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
          'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 60
BATCH = 512
rng = np.random.default_rng(0)

idx = [r for r in csv.DictReader((DS / 'index.csv').open(encoding='utf-8'))]
X = np.load(FEAT).astype(np.float32)
Kc = np.log1p(np.nan_to_num(np.load(CTX), nan=0.0)).astype(np.float32)
cls = np.array([r['class'] for r in idx]); ports = np.array([r['port'] for r in idx])
known = np.isin(cls, KNOWN8)
y = np.full(len(idx), -1)
for j, c in enumerate(KNOWN8):
    y[cls == c] = j
PORT_U = sorted(set(ports.tolist()))
port_of = {p: i for i, p in enumerate(PORT_U)}
pid = np.array([port_of[p] for p in ports])

D_CHIP, D_CTX = X.shape[1], 64
print('chips %d | known %d | unknown %d | device %s' % (len(idx), int(known.sum()), int((~known).sum()), DEV))


def make_model(drop_novelty_gate=False):
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = KOSR(n_known=8, d_chip=D_CHIP, d_graph=1 + Kc.shape[1], d_ctx=D_CTX, n_blocks=2)
            self.drop = drop_novelty_gate

        def forward(self, z, nf, adj, with_knowledge=True):
            o = self.k(z, nf, adj, with_knowledge=with_knowledge)
            if with_knowledge and self.drop:
                o['logits'] = o['logits_v'] + o['kappa'].unsqueeze(1) * o['logits_k']
            return o
    return Wrap().to(DEV)


def node_features(rows_pid, rows_ctx):
    nf = np.zeros((len(rows_pid), 1, 1 + Kc.shape[1]), np.float32)
    for i, (p, c) in enumerate(zip(rows_pid, rows_ctx)):
        nf[i, 0, 0] = 1.0
        nf[i, 0, 1:] = c
    return torch.tensor(nf, device=DEV)


def run_arm(drop, tr, te):
    torch.manual_seed(0)
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    model = make_model(drop_novelty_gate=drop)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ztr = torch.tensor(Xtr, device=DEV); ytr = torch.tensor(y[tr], device=DEV)
    nftr = node_features(pid[tr], Kc[tr]); adj_tr = torch.ones(len(tr), 1, 1, device=DEV)
    last = None
    for ep in range(EPOCHS):
        perm = torch.randperm(len(tr), device=DEV)
        tot, nb = 0.0, 0
        for i in range(0, len(tr), BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(ztr[j], nftr[j], adj_tr[j])
            loss, _ = kosr_loss(out, ytr[j])
            loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        last = tot / max(1, nb)
    print('      [train] 末轮 CE %.4f' % last, flush=True)
    model.eval()
    with torch.no_grad():
        o = model(torch.tensor(Xte, device=DEV), node_features(pid[te], Kc[te]),
                  torch.ones(len(te), 1, 1, device=DEV))
        p = softmax(o['logits'].cpu().numpy(), axis=1)
        kv = softmax(o['logits_v'].cpu().numpy(), axis=1)
    return p, kv


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


acc = {a: [] for a in ('visual', 'kosr', 'kosr_nogate')}
liqshare = {a: [] for a in ('visual', 'kosr', 'kosr_nogate')}
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
for p in PORT_U:
    tr = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(tr) < 200 or len(te) < 20:
        continue
    pr, pv = run_arm(False, tr, te)
    pn, pvn = run_arm(True, tr, te)
    tk = known[te]; ty = y[te]
    acc['visual'].append(ba(ty[tk], pv[tk].argmax(1)))
    acc['kosr'].append(ba(ty[tk], pr[tk].argmax(1)))
    acc['kosr_nogate'].append(ba(ty[tk], pn[tk].argmax(1)))
    for a, pp in (('visual', pv), ('kosr', pr), ('kosr_nogate', pn)):
        liqshare[a].append(float(np.isin(pp[~tk].argmax(1), list(LIQ)).mean()) if (~tk).sum() else float('nan'))
    print('%-16s known n=%5d 未知 n=%5d | 视觉 %.3f  KOSR %.3f  无门控 %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), acc['visual'][-1], acc['kosr'][-1], acc['kosr_nogate'][-1]), flush=True)

print('')
print('（视觉列 = 同一网络的 head_v 路径，非 ridge；ridge 同特征参考 0.215）')
for a in acc:
    print('%-12s 已知类 BA 均值 %.4f  未知片液货族占比 %.4f' % (
        a, float(np.nanmean(acc[a])), float(np.nanmean(liqshare[a]))))

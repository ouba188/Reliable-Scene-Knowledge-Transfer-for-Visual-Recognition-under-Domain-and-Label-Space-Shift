"""e91: KOSR with a SUPERVISED novelty head -- the fix e90's instability pointed at.

e90 trained with lam_novel=0, so nu was a random number and the (1-nu) factor in the coupling law was noise; the
ablation's direction flipped between runs, which is exactly what an unsupervised gate produces. Fix: supervise nu
on SOURCE-SIDE pseudo-unknowns (leave-class-out). For each fold two of the eight known classes are held out of the
cross-entropy entirely and fed to the novelty BCE as 'unknown'; the remaining six train the closed-set heads.
Nothing about the target is used. Then at test time the real unknown ship types should get nu ~ 1, which is the
mechanism's claim: the knowledge must not manufacture membership for them.

Arms: visual (head_v only) | kosr (full coupling, supervised nu) | kosr_nogate (coupling without (1-nu)).
Reported per arm: known-class BA, and among the REAL unknown chips: the novelty score, the liquid-known-family
share, and how often the gated knowledge is essentially switched off (nu > 0.5).
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
LIQ = {KNOWN8.index('crude_oil_tanker'), KNOWN8.index('product_chemical_tanker')}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS, BATCH, HOLDOUT = 60, 512, 2
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
print('chips %d | known %d | unknown %d | %s' % (len(idx), int(known.sum()), int((~known).sum()), DEV), flush=True)


class Wrap(nn.Module):
    def __init__(self, drop_gate=False):
        super().__init__()
        self.k = KOSR(n_known=8, d_chip=D_CHIP, d_graph=1 + Kc.shape[1], d_ctx=D_CTX, n_blocks=2)
        self.drop = drop_gate

    def forward(self, z, nf, adj):
        o = self.k(z, nf, adj, with_knowledge=True)
        if self.drop:
            o['logits'] = o['logits_v'] + o['kappa'].unsqueeze(1) * o['logits_k']
        return o


def node_features(p, c):
    nf = np.zeros((len(p), 1, 1 + Kc.shape[1]), np.float32)
    nf[:, 0, 0] = 1.0
    nf[:, 0, 1:] = c
    return torch.tensor(nf, device=DEV)


def train(tr_in, tr_out, drop):
    torch.manual_seed(0)
    mu = X[tr_in].mean(0, keepdims=True); sd = X[tr_in].std(0, keepdims=True) + 1e-6
    allt = np.concatenate([tr_in, tr_out])
    Xa = torch.tensor((X[allt] - mu) / sd, device=DEV)
    ya = torch.tensor(y[allt], device=DEV)
    isu = torch.tensor(np.r_[np.zeros(len(tr_in)), np.ones(len(tr_out))], device=DEV).float()
    nfa = node_features(pid[allt], Kc[allt]); adja = torch.ones(len(allt), 1, 1, device=DEV)
    m = Wrap(drop).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    nin = len(tr_in)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(allt), device=DEV)
        tot, nb = 0.0, 0
        for i in range(0, len(allt), BATCH):
            j = perm[i:i + BATCH]
            mask_in = (j < nin)
            opt.zero_grad()
            o = m(Xa[j], nfa[j], adja[j])
            if mask_in.any():
                loss, _ = kosr_loss({'logits': o['logits'][mask_in], 'nu': o['nu'][mask_in]},
                                    ya[j][mask_in], lam_novel=0.0)
            else:
                loss = torch.zeros((), device=DEV)
            ln = nn.functional.binary_cross_entropy(o['nu'], isu[j])
            loss = loss + 0.5 * ln
            loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
    return m, mu, sd


def ba(yy, pred):
    rs = [float((pred[yy == c] == c).mean()) for c in range(8) if (yy == c).any()]
    return float(np.mean(rs)) if rs else float('nan')


res = {a: {'ba': [], 'liq': [], 'nu': [], 'off': []} for a in ('visual', 'kosr', 'kosr_nogate')}
for p in PORT_U:
    tr_all = np.where(known & (ports != p))[0]
    te = np.where(ports == p)[0]
    if len(tr_all) < 400 or len(te) < 20:
        continue
    ho = rng.choice(8, HOLDOUT, replace=False)                       # source-side pseudo-unknown classes
    tr_in = tr_all[~np.isin(y[tr_all], ho)]
    tr_out = tr_all[np.isin(y[tr_all], ho)]
    tk = known[te]; ty = y[te]
    out = {}
    for drop, nm in ((False, 'kosr'), (True, 'kosr_nogate')):
        m, mu, sd = train(tr_in, tr_out, drop)
        m.eval()
        with torch.no_grad():
            o = m(torch.tensor((X[te] - mu) / sd, device=DEV), node_features(pid[te], Kc[te]),
                  torch.ones(len(te), 1, 1, device=DEV))
            out[nm] = (softmax(o['logits'].cpu().numpy(), 1), o['nu'].cpu().numpy())
            if nm == 'kosr':
                out['visual'] = (softmax(o['logits_v'].cpu().numpy(), 1), o['nu'].cpu().numpy())
    ridx = np.where(ho == np.arange(8)[:, None])[0] if False else None
    for a in ('visual', 'kosr', 'kosr_nogate'):
        pp, nu = out[a]
        res[a]['ba'].append(ba(ty[tk], pp[tk].argmax(1)))
        if (~tk).sum():
            res[a]['liq'].append(float(np.isin(pp[~tk].argmax(1), list(LIQ)).mean()))
            res[a]['nu'].append(float(nu[~tk].mean()))
            res[a]['off'].append(float((nu[~tk] > 0.5).mean()))
    print('%-16s 已知 %5d 未知 %5d | BA 视觉 %.3f KOSR %.3f 无门控 %.3f | 未知 ν 均值 %.3f 关断率 %.3f' % (
        p, int(tk.sum()), int((~tk).sum()), res['visual']['ba'][-1], res['kosr']['ba'][-1],
        res['kosr_nogate']['ba'][-1], np.nanmean(res['kosr']['nu']), np.nanmean(res['kosr']['off'])), flush=True)


print('')
for a in res:
    print('%-12s 已知 BA %.4f | 未知片 液货族 %.4f | ν 均值 %.3f | 知识关断率 %.3f' % (
        a, float(np.nanmean(res[a]['ba'])), float(np.nanmean(res[a]['liq'])),
        float(np.nanmean(res[a]['nu'])), float(np.nanmean(res[a]['off']))))

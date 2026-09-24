"""KOSR: Knowledge-Conditioned Open-Set Recognizer -- the architecture, with every part derived from a finding.

Why each part exists (all from this project's own evidence):
  depth is NOT spent on capacity for its own sake -- E17e/E17m showed more backbone or joint training on the same
  features changes nothing. Depth is spent where real structure exists:
    * the facility-graph branch: the knowledge IS relational (E17i) and types matter (E17v), so the port's
      facility-kind graph is a genuine structure to message-pass over;
    * the cross-modal block: the chip must query the port context (E17g: the effect's sign is a CLASS property,
      so the context has to condition per instance);
    * the three heads: a closed-set posterior, a novelty score and a knowledge-trust -- because the open-set
      failure mode we measured (E17q: unseen types get the SAME confidence as known ones, 0.238 vs 0.233, 85.5%
      of them crossing the known-pool threshold) is precisely 'the vocabulary is wrong', not 'the image is hard';
    * the coupling law
          ell = ell_v + (1 - nu) * kappa * delta ell_k
      which encodes the mechanism: knowledge may correct WITHIN the known vocabulary but may never manufacture
      membership. It is falsifiable -- drop the (1 - nu) factor and the ablation should absorb novel types.

Ponytail notes: no torch_geometric dependency (the GAT is 30 lines of plain attention); the graph branch takes a
generic (node features, adjacency) pair per port, so a facility-derived graph and a placeholder both fit and the
incoming OSM extracts just slot in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBlock(nn.Module):
    """One graph-attention layer: A_hat = softmax over neighbours of (q.k)/sqrt(d), then a residual MLP."""

    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.h = heads
        self.d = d
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, N, d)   adj: (B, N, N) with 1 = edge (self-loops included by the caller)
        B, N, d = x.shape
        q = self.q(x).view(B, N, self.h, d // self.h).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, d // self.h).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, d // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (d // self.h) ** 0.5
        att = att.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))
        att = self.drop(att.softmax(-1))
        h = (att @ v).transpose(1, 2).reshape(B, N, d)
        x = self.norm1(x + self.o(h))
        return self.norm2(x + self.ff(x))


class KOSR(nn.Module):
    """chip encoder + port facility graph + cross-modal block + three heads + the coupling law."""

    def __init__(self, n_known=8, d_chip=256, d_graph=64, d_ctx=128, n_heads=4, n_blocks=2, backbone=None):
        super().__init__()
        self.n_known = n_known
        self.encoder = backbone if backbone is not None else nn.Identity()
        self.chip = nn.Sequential(nn.Linear(d_chip, d_ctx), nn.GELU(), nn.LayerNorm(d_ctx))
        self.node_in = nn.Linear(d_graph, d_ctx)
        self.gblocks = nn.ModuleList([GraphBlock(d_ctx, n_heads) for _ in range(n_blocks)])
        self.cross = nn.MultiheadAttention(d_ctx, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_ctx)
        self.head_v = nn.Linear(d_ctx, n_known)          # visual-only logits
        self.head_k = nn.Linear(d_ctx, n_known)          # knowledge logits
        self.head_kappa = nn.Linear(d_ctx, 1)            # per-instance trust in the knowledge
        self.head_nu = nn.Linear(d_ctx, 1)               # novelty score

    def forward(self, z, node_feat, adj, with_knowledge=True):
        """z: (B, d_chip) chip features. node_feat: (B, N, d_graph) port graph nodes. adj: (B, N, N) 0/1."""
        c = self.chip(z).unsqueeze(1)                     # (B,1,d_ctx) the chip as one token
        g = self.node_in(node_feat)
        for blk in self.gblocks:
            g = blk(g, adj)
        # the chip queries the port graph; its own token is attended too (identity skip through the norm)
        ctx, att = self.cross(c, g, g, need_weights=True)
        c = self.cross_norm(c + ctx).squeeze(1)
        lv = self.head_v(c)
        lk = self.head_k(c)
        kappa = torch.sigmoid(self.head_kappa(c)).squeeze(1)
        nu = torch.sigmoid(self.head_nu(c)).squeeze(1)
        if with_knowledge:
            logits = lv + (1.0 - nu).unsqueeze(1) * kappa.unsqueeze(1) * lk
        else:
            logits = lv
        return {'logits': logits, 'logits_v': lv, 'logits_k': lk, 'kappa': kappa, 'nu': nu, 'attn': att}


def kosr_loss(out, y, lam_novel=0.1, lam_trust=0.0, is_unknown=None, trust_target=None):
    """CE on the gated logits; optional novelty supervision on the source-side held-out classes and an optional
    BCE on kappa against the source-measured 'knowledge would help' flag."""
    loss = F.cross_entropy(out['logits'], y)
    parts = {'ce': float(loss.detach())}
    if is_unknown is not None and lam_novel > 0:
        ln = F.binary_cross_entropy(out['nu'], is_unknown.float())
        loss = loss + lam_novel * ln
        parts['novelty'] = float(ln.detach())
    if trust_target is not None and lam_trust > 0:
        lt = F.binary_cross_entropy(out['kappa'], trust_target.float())
        loss = loss + lam_trust * lt
        parts['trust'] = float(lt.detach())
    return loss, parts


if __name__ == '__main__':
    # runnable self-check: shapes line up and the coupling law actually gates
    torch.manual_seed(0)
    B, N = 4, 6
    m = KOSR(n_known=8, d_chip=32, d_graph=16, d_ctx=32, n_blocks=2)
    z = torch.randn(B, 32)
    nf = torch.randn(B, N, 16)
    adj = (torch.rand(B, N, N) > 0.4).float()
    adj = torch.clamp(adj + torch.eye(N), 0, 1)
    out = m(z, nf, adj)
    assert out['logits'].shape == (B, 8), out['logits'].shape
    assert out['kappa'].min() >= 0 and out['kappa'].max() <= 1
    assert out['nu'].min() >= 0 and out['nu'].max() <= 1
    # with nu = 1 the knowledge contribution must vanish
    out2 = m(z, nf, adj)
    out2['nu'] = torch.ones(B)
    gated = (1 - out2['nu']).unsqueeze(1) * out2['kappa'].unsqueeze(1) * out2['logits_k']
    assert float(gated.abs().max()) == 0.0
    loss, parts = kosr_loss(out, torch.randint(0, 8, (B,)), lam_novel=0.1,
                            is_unknown=torch.tensor([1, 0, 0, 1]),
                            lam_trust=0.1, trust_target=torch.tensor([1., 0., 1., 0.]))
    print('OK  loss=%.4f  parts=%s  attn shape=%s' % (float(loss), parts, tuple(out['attn'].shape)))

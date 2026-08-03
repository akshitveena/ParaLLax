"""
sdae_prm.py — Step-structured Denoising Autoencoder + PRM head.

Input is a SEQUENCE of step embeddings (one vector per reasoning step). A Transformer
mixes them (so step i can be judged against step i-1), which is the bottleneck. Three
outputs:
  * decoder    -> reconstruct the CLEAN step-embedding sequence   (denoising AE)
  * prm_head   -> per-step error logit                            (PRM, ProcessBench labels)
  * chain_head -> A/B from attention-pooled step-codes            (chain validity)

No cross-step pooling happens before the sequence model, so relational validity — the
thing mean-pooling averaged away (pooled ceiling ~0.29 f1_B) — is representable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, maxlen: int = 512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                       # x: (B, T, d)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class StepSDAE_PRM(nn.Module):
    def __init__(self, in_dim: int = 384, d_model: int = 256, nhead: int = 4,
                 nlayers: int = 2, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, in_dim))
        self.prm_head = nn.Linear(d_model, 1)
        self.attn = nn.Linear(d_model, 1)
        self.chain_head = nn.Linear(d_model, 1)

    def encode(self, steps, pad_mask, corrupt_mask=None):
        x = self.proj(steps)
        if corrupt_mask is not None:
            x = torch.where(corrupt_mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), x)
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=pad_mask)      # (B, T, d)

    def forward(self, steps, pad_mask, corrupt_mask=None):
        h = self.encode(steps, pad_mask, corrupt_mask)
        recon = self.decoder(h)                                    # (B, T, in_dim)
        prm_logit = self.prm_head(h).squeeze(-1)                   # (B, T)
        a = self.attn(h).squeeze(-1).masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(1)                                    # (B, d) attention-pooled
        chain_logit = self.chain_head(pooled).squeeze(-1)          # (B,)
        return recon, prm_logit, chain_logit, pooled


def losses(recon, steps, prm_logit, step_labels, chain_logit, chain_label,
           pad_mask, corrupt_mask, lam_prm=1.0, lam_ab=1.0, heads="prm_chain"):
    """Total = L_denoise (+ PRM/chain when active).

    `heads` selects which supervised heads contribute: "prm_chain" (default, the full
    model — identical to previous behaviour), "prm", "chain", or "none" (recon-only,
    Variant R of the Phase-1c content probe). L_denoise is always on.
    """
    valid = ~pad_mask
    # denoising: reconstruct the CLEAN step embeddings (cosine) over valid steps
    cos = (F.normalize(recon, dim=-1) * F.normalize(steps, dim=-1)).sum(-1)   # (B, T)
    l_denoise = (1.0 - cos)[valid].mean()
    # PRM: per-step error BCE on labeled steps only (step_labels in {0,1}; -1 ignored)
    prm_mask = (step_labels >= 0) & valid
    if prm_mask.any():
        l_prm = F.binary_cross_entropy_with_logits(prm_logit[prm_mask],
                                                    step_labels[prm_mask].float())
    else:
        l_prm = torch.zeros((), device=recon.device)
    l_ab = F.binary_cross_entropy_with_logits(chain_logit, chain_label.float())
    total = l_denoise
    if "prm" in heads:
        total = total + lam_prm * l_prm
    if "chain" in heads:
        total = total + lam_ab * l_ab
    return total, l_denoise.detach(), l_prm.detach(), l_ab.detach()

"""
ridae.py — the RiDAE model: encoder + bottleneck + decoder + losses.

Architecture (Roadmap "Corrected Edition"):

    corrupted text
        -> [ENCODER]  all-MiniLM-L6-v2 (fine-tuned, mean-pooled, 384-d)
        -> [BOTTLENECK FFN]  384 -> 256 -> 128 -> 64   = z  (interpretable code)
        -> [DECODER]  64 -> 128 -> 256 -> 384          = reconstructed embedding

Losses (all three train simultaneously):
    L_reconstruct : 1 - cos(decoder(z_corrupted), encoder(original))   (TSDAE-style)
    L_MNR         : in-batch multiple-negatives ranking on (z_corr, z_orig)
    L_triplet     : push Type A and Type B apart in z-space (margin 0.5)
    L_total = L_reconstruct + L_MNR + lambda * L_triplet     (lambda = 0.3)

The encoder is FINE-TUNED (not frozen): the reconstruction objective must reshape
how the encoder represents reasoning, not just learn a projection on frozen features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def auto_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
class BottleneckFFN(nn.Module):
    """384 -> 256 -> 128 -> 64, LayerNorm + GELU + Dropout between stages.

    LayerNorm prevents the bottleneck from collapsing to near-zero vectors on
    short inputs; GELU (vs ReLU) lets informative small-negative activations pass.
    """

    def __init__(self, in_dim: int = 384, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, latent_dim), nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReasoningDecoder(nn.Module):
    """Mirror of the bottleneck, expanding 64 -> 128 -> 256 -> 384.

    Produces a 384-d embedding (NOT text) compared to the original via cosine.
    Kept for reconstruction loss AND novel-reasoning generation via interpolation.
    """

    def __init__(self, latent_dim: int = 64, out_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TripletContrastiveLoss(nn.Module):
    """Triplet margin loss (Euclidean, margin 0.5) over z codes."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative) -> torch.Tensor:
        return self.loss(anchor, positive, negative)


def multiple_negatives_ranking_loss(z_anchor: torch.Tensor,
                                    z_positive: torch.Tensor,
                                    temperature: float = 0.07) -> torch.Tensor:
    """In-batch MNR loss on the 64-d codes.

    Each anchor (corrupted code) must rank its own positive (original code) above
    every other positive in the batch. Larger batches -> more negatives -> richer
    signal. This is what gives z its inter-candidate similarity geometry.
    """
    a = F.normalize(z_anchor, dim=-1)
    p = F.normalize(z_positive, dim=-1)
    scores = (a @ p.t()) / temperature              # (B, B)
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores, labels)


# --------------------------------------------------------------------------- #
class RiDAE(nn.Module):
    def __init__(self,
                 encoder_name: str = DEFAULT_ENCODER,
                 latent_dim: int = 64,
                 dropout: float = 0.1,
                 device: str | None = None,
                 mnr_temperature: float = 0.07,
                 triplet_margin: float = 0.5):
        super().__init__()
        self.device = auto_device(device)
        self.encoder_name = encoder_name
        self.latent_dim = latent_dim
        self.mnr_temperature = mnr_temperature

        self.st = SentenceTransformer(encoder_name, device=str(self.device))
        # Locate the underlying HF transformer + tokenizer for grad-tracking encode.
        self._transformer = self.st[0]            # models.Transformer
        self.embed_dim = self.st.get_sentence_embedding_dimension()

        self.bottleneck = BottleneckFFN(self.embed_dim, latent_dim, dropout)
        self.decoder = ReasoningDecoder(latent_dim, self.embed_dim, dropout)
        self.triplet = TripletContrastiveLoss(margin=triplet_margin)

        self.to(self.device)

    # ----- encoding -------------------------------------------------------- #
    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def _encode_with_grad(self, texts: list[str], max_length: int = 384) -> torch.Tensor:
        """Mean-pooled, L2-normalised 384-d embeddings WITH gradient tracking."""
        feats = self._transformer.tokenize(texts)
        feats = {k: v.to(self.device) for k, v in feats.items()
                 if isinstance(v, torch.Tensor)}
        out = self._transformer.auto_model(
            input_ids=feats["input_ids"],
            attention_mask=feats["attention_mask"],
        )
        token_emb = out[0]                              # (B, T, 384)
        pooled = self._mean_pool(token_emb, feats["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """No-grad encode to z-space (N, latent_dim) for analysis."""
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            emb = self._encode_with_grad(chunk)
            z = self.bottleneck(emb)
            out.append(z.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.latent_dim))

    # ----- forward + losses ----------------------------------------------- #
    def forward(self, corrupted_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._encode_with_grad(corrupted_texts)
        z = self.bottleneck(emb)
        reconstructed = self.decoder(z)
        return z, reconstructed

    def reconstruction_loss(self, corrupted_texts: list[str],
                            original_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (L_reconstruct, z_corrupted). z is reused for MNR."""
        z, reconstructed = self.forward(corrupted_texts)
        with torch.no_grad():
            target = self._encode_with_grad(original_texts).detach()
        cos = F.cosine_similarity(reconstructed, target, dim=1)
        return (1.0 - cos.mean()), z

    def mnr_loss(self, corrupted_texts: list[str], original_texts: list[str]) -> torch.Tensor:
        z_corr = self.bottleneck(self._encode_with_grad(corrupted_texts))
        z_orig = self.bottleneck(self._encode_with_grad(original_texts))
        return multiple_negatives_ranking_loss(z_corr, z_orig, self.mnr_temperature)

    def contrastive_loss_from_pairs(self, type_a_texts: list[str],
                                    type_b_texts: list[str]) -> torch.Tensor:
        """Triplet loss: split A in half for anchor/positive, B as negative.

        Returns 0 (no grad) if there are too few pairs to form a triplet batch.
        """
        n = min(len(type_a_texts), len(type_b_texts))
        if n < 2:
            return torch.zeros((), device=self.device)
        half = n // 2
        if half < 1:
            return torch.zeros((), device=self.device)

        z_a = self.bottleneck(self._encode_with_grad(type_a_texts[:n]))
        z_b = self.bottleneck(self._encode_with_grad(type_b_texts[:n]))
        anchor = z_a[:half]
        positive = z_a[half:2 * half]
        negative = z_b[:half]
        m = min(anchor.size(0), positive.size(0), negative.size(0))
        if m < 1:
            return torch.zeros((), device=self.device)
        return self.triplet(anchor[:m], positive[:m], negative[:m])

    # ----- persistence ----------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder_name": self.encoder_name,
            "latent_dim": self.latent_dim,
            "encoder_state": self.st.state_dict(),
            "bottleneck_state": self.bottleneck.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "mnr_temperature": self.mnr_temperature,
        }, path)

    @classmethod
    def load(cls, path: str | Path, encoder_name: str | None = None,
             device: str | None = None) -> "RiDAE":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(
            encoder_name=encoder_name or ckpt["encoder_name"],
            latent_dim=ckpt.get("latent_dim", 64),
            device=device,
            mnr_temperature=ckpt.get("mnr_temperature", 0.07),
        )
        model.st.load_state_dict(ckpt["encoder_state"])
        model.bottleneck.load_state_dict(ckpt["bottleneck_state"])
        model.decoder.load_state_dict(ckpt["decoder_state"])
        model.to(model.device)
        return model


if __name__ == "__main__":
    m = RiDAE(device="cpu")
    z, rec = m.forward(["1. add things. 2. multiply. 3. answer is \\boxed{18}."])
    print("z", z.shape, "reconstructed", rec.shape, "device", m.device)
    print("encode ->", m.encode(["a reasoning chain", "another chain"]).shape)

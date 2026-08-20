"""
E2 — Factored architecture: content-preserving encoder + non-competing validity readout.
(Reviewer ask #2 / Appendix N follow-up.)

This is a TEST, not a coronation. Appendix N flags that a fixed content-preserving latent may not
supply the separability the shared-bottleneck e2e model creates by reshaping the encoder:
    frozen recon-only latent (Variant R): validity AUC 0.693, f1B 0.416
    shared e2e (Variant F-e2e):           validity AUC 0.741, controlled f1B 0.591
So the factored design must be shown to reach content Recall@1 ≈ 0.996 AND f1B ≥ 0.591 before it
earns the main text. See PLAN.md "E2 decision rule".

Three regimes trained on identical data/seeds, one readout protocol, one eval axis:
  (A) FROZEN recon-only encoder + MLP readout.  Gradients from validity NEVER reach the encoder.
  (B) recon encoder + a SEPARATE adapter (LoRA-style) trained on validity only, structurally
      isolated from the reconstruction decoder — lets validity reshape features WITHOUT touching the
      reconstruction bottleneck. The honest attempt at "e2e validity, preserved content".
  (C) shared-bottleneck e2e (reproduce Variant F-e2e) — the baseline to beat.

Report per regime: Recall@1 (content), controlled f1B, validity AUC. 5 seeds, paired bootstrap.
"""
from __future__ import annotations
import argparse, json
from typing import Callable
import numpy as np
import torch
import torch.nn as nn

# ============================== ADAPTER ==============================
# Reuse the paper's Step-SDAE components. These stubs describe exactly what to return.

def load_steps_and_labels(split: str):
    """split in {'train','val'}. Returns:
       step_embeds : list over candidates of Tensor[N_i, 384]  (all-MiniLM-L6-v2 clean step embeds)
       chain_label : LongTensor[C]                             (1 == Type A / sound, 0 == Type B)
       step_labels : list over candidates of LongTensor[N_i]   (per-step validity, -1 where unlabelled)
       confounds   : FloatTensor[C, 4]                         (length, latex_density, n_steps, dataset-id)
    """
    raise NotImplementedError

def encoder_recon_only(seed: int) -> nn.Module:
    """Your Step-SDAE encoder+bottleneck trained with L_denoise ONLY (Variant R). Should already give
    Recall@1 ≈ 0.996. forward(step_embeds)->step_codes Tensor[N,256]. This is regime (A)/(B)'s encoder."""
    raise NotImplementedError

def build_shared_e2e(seed: int) -> nn.Module:
    """Full Step-SDAE (denoise+PRM+chain heads, shared bottleneck), unfrozen — Variant F-e2e (C)."""
    raise NotImplementedError

def residualize_fit(codes: np.ndarray, confounds: np.ndarray):
    """Fit the paper's LINEAR 4-confound residualizer on TRAIN codes. Returns an apply() closure."""
    raise NotImplementedError

def recall_at_1(encoder: nn.Module, val_step_embeds) -> float:
    """Content-preservation probe from Table 6: decode step-codes, retrieve against all held-out
    clean targets, fraction whose rank-1 retrieval is the true source. Recon-only ≈ 0.996."""
    raise NotImplementedError

def f1b_and_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Type-B f1 (positive class = Type B == label 0) at the paper's threshold, and ROC-AUC.
    Reuse the repo's exact f1B implementation so numbers are comparable to Table 1/7."""
    raise NotImplementedError
# ============================ END ADAPTER ============================


# ---- attention-pooled chain readout used by regimes (A) and (B); does NOT feed the encoder in (A) ----
class ValidityReadout(nn.Module):
    def __init__(self, d_in: int = 256, d_hidden: int = 256):
        super().__init__()
        self.attn = nn.Linear(d_in, 1)
        self.mlp = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, step_codes: torch.Tensor) -> torch.Tensor:  # [N, d] -> scalar logit
        w = torch.softmax(self.attn(step_codes), dim=0)            # attention pool over steps
        pooled = (w * step_codes).sum(0)
        return self.mlp(pooled).squeeze(-1)


class Adapter(nn.Module):
    """Regime (B): a low-rank residual adapter inserted on the encoder OUTPUT, trained on validity
    only. It reshapes features for the readout without any gradient path into the reconstruction
    decoder (the decoder reads the un-adapted codes). This is the 'non-competing' pathway."""
    def __init__(self, d: int = 256, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.up.weight)  # start as identity

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return codes + self.up(self.down(codes))


def _train_readout(encoder, adapter, readout, data, *, epochs=30, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    step_embeds, chain_label, _, _ = data
    params = list(readout.parameters()) + ([] if adapter is None else list(adapter.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()
    encoder.eval()
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for emb, y in zip(step_embeds, chain_label):
            with torch.no_grad():                       # (A)/(B): encoder is FROZEN — key invariant
                codes = encoder(emb)
            if adapter is not None:
                codes = adapter(codes)                  # (B): adapter IS trained, encoder is not
            logit = readout(codes)
            loss = loss + bce(logit, y.float())
        (loss / len(chain_label)).backward()
        opt.step()
    return readout


def _score(encoder, adapter, readout, data):
    step_embeds, _, _, _ = data
    encoder.eval(); readout.eval()
    out = []
    with torch.no_grad():
        for emb in step_embeds:
            codes = encoder(emb)
            if adapter is not None:
                codes = adapter(codes)
            out.append(torch.sigmoid(readout(codes)).item())
    return np.array(out)


def run(regime: str, seeds=range(5)) -> dict:
    train, val = load_steps_and_labels("train"), load_steps_and_labels("val")
    val_labels = val[1].numpy()
    val_conf = val[3].numpy()
    results = {"recall_at_1": [], "ctrl_f1b": [], "auc": []}
    for s in seeds:
        if regime in ("A", "B"):
            enc = encoder_recon_only(s)
            adapter = Adapter().train() if regime == "B" else None
            readout = ValidityReadout()
            _train_readout(enc, adapter, readout, train, seed=s)
            content = recall_at_1(enc, val[0])          # (A)/(B) share the recon-only encoder
            raw_scores = _score(enc, adapter, readout, val)
        elif regime == "C":
            model = build_shared_e2e(s)                 # trains all heads jointly (your existing loop)
            content = recall_at_1(model, val[0])        # expected to collapse (~0.007 e2e)
            raw_scores = model.chain_scores(val[0])     # your existing chain-head inference
        else:
            raise ValueError(regime)
        # confound-controlled evaluation: residualize scores' representation, then f1B/AUC
        apply_resid = residualize_fit(np.asarray(raw_scores).reshape(-1, 1), None)  # see note below
        ctrl_scores = apply_resid(np.asarray(raw_scores).reshape(-1, 1)).ravel()
        f1b, auc = f1b_and_auc(ctrl_scores, val_labels)
        results["recall_at_1"].append(content)
        results["ctrl_f1b"].append(f1b)
        results["auc"].append(auc)
    agg = {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}
    return {"regime": regime, "per_seed": results, "mean_std": agg,
            "targets": {"recall_at_1": 0.996, "ctrl_f1b": 0.591, "auc": 0.741},
            "note": "NOTE: residualize the READOUT'S INPUT representation (step-codes / pooled vector), "
                    "not the 1-D score, to match Table 1's protocol. The reshape above is a placeholder "
                    "for the adapter's residualize_fit(codes, confounds). Pass real codes+confounds."}


DECISION = """DECISION RULE (feeds paper/factored_architecture_maintext.md):
  best of {A,B} reaches Recall@1 ~0.996 AND ctrl_f1b >= 0.591  -> promote to main Method (ask #2 met).
  content preserved but auc ~0.69 / f1b < 0.591                 -> present as content-preserving variant
                                                                    WITH a stated validity cost.
  fails both                                                    -> stays an appendix direction; do not move.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--out", default="e2_results.json")
    args = ap.parse_args()
    regimes = ["A", "B", "C"] if args.regime == "all" else [args.regime]
    res = {r: run(r) for r in regimes}
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n" + DECISION)

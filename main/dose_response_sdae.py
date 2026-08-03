"""
dose_response_sdae.py — toxicology of the STEP-STRUCTURED model (Phase-4 second half).

Corrupts step-sequences at rising doses and measures, per dose:
  * reconstruction error (1 - cos of decoded vs the clean frozen step embeddings) — can the
    denoising AE still undo the damage?
  * chain A/B AUC — does the VALIDITY signal survive the damage?

Four operators (length-preserving, so reconstruction stays aligned):
  word_delete   — delete a fraction of words per step, re-encode (text-level, TSDAE-style)
  step_mask     — mask a fraction of step vectors (the training corruption; necrosis analog)
  step_shuffle  — shuffle a fraction of step positions (breaks sequencing; demyelination)
  vector_noise  — additive Gaussian noise on step vectors (oxidative analog)

The knee of each reconstruction curve = that damage's breakdown point (the manifold radius
along that direction); the A/B-AUC curve shows how robust validity is to it.

    python main/dose_response_sdae.py --checkpoint checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt \
        --n 150 --out_dir outputs_tox --device cpu
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdae_prm import StepSDAE_PRM
from ridae import RiDAE
from train_sdae_e2e import split_recs, corrupt_text

OPS = ["word_delete", "step_mask", "step_shuffle", "vector_noise"]
DOSES = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/step_cache.pt")
    ap.add_argument("--checkpoint", default="checkpoints/checkpoints_sdae_e2e/sdae_e2e_best.pt")
    ap.add_argument("--n", type=int, default=150, help="held-out candidates sampled for the sweep")
    ap.add_argument("--out_dir", default="outputs_tox")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    dev = torch.device(args.device)
    recs = torch.load(args.cache, weights_only=False)
    _, va = split_recs(recs)
    va = va[: args.n]
    ck = torch.load(args.checkpoint, map_location=args.device)
    enc = RiDAE(device=args.device); enc.st.load_state_dict(ck["enc"]); enc.eval()
    sdae = StepSDAE_PRM().to(dev); sdae.load_state_dict(ck["sdae"]); sdae.eval()

    # precompute clean encodings + frozen targets + labels
    Xc, tgt, texts, y = [], [], [], []
    with torch.no_grad():
        for r in va:
            Xc.append(enc._encode_with_grad(r["steps_text"]))          # (n, 384) trained-encoder
            tgt.append(torch.from_numpy(r["steps_emb"]).float().to(dev))
            texts.append(r["steps_text"])
            y.append(0 if r["chain"] == "A" else 1)
    y = np.array(y)

    rows = []
    for op in OPS:
        for s in DOSES:
            errs, pbs = [], []
            for i in range(len(va)):
                X0, t = Xc[i], tgt[i]; n = X0.size(0)
                cm = None
                if op == "word_delete":
                    rng = random.Random(1000 + i)
                    ct = [corrupt_text(w, rng, s) for w in texts[i]]
                    with torch.no_grad():
                        X = enc._encode_with_grad(ct)
                    if X.size(0) != n:                                  # segmentation stable; guard anyway
                        X = X[:n] if X.size(0) > n else torch.cat([X, X0[X.size(0):]], 0)
                elif op == "step_mask":
                    X = X0.clone()
                    k = int(n * s)
                    if k:
                        cm = torch.zeros(1, n, dtype=torch.bool, device=dev)
                        cm[0, torch.randperm(n)[:k]] = True
                elif op == "step_shuffle":
                    X = X0.clone(); k = int(n * s)
                    if k >= 2:
                        idx = torch.randperm(n)[:k]
                        X[idx] = X0[idx[torch.randperm(k)]]
                else:  # vector_noise
                    X = X0 + s * X0.std() * torch.randn_like(X0)
                with torch.no_grad():
                    recon, prm, cl, _ = sdae(X.unsqueeze(0),
                                             torch.zeros(1, n, dtype=torch.bool, device=dev), cm)
                errs.append((1 - F.cosine_similarity(recon[0], t, dim=-1)).mean().item())
                pbs.append(torch.sigmoid(cl).item())
            auc = roc_auc_score(y, pbs) if len(set(y)) > 1 else float("nan")
            rows.append((op, s, float(np.mean(errs)), float(auc)))
            print(f"  {op:14} dose={s:.2f}  recon_err={np.mean(errs):.3f}  chainAUC={auc:.3f}", flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "dose_response_sdae.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["operator", "dose", "recon_err", "chain_auc"]); w.writerows(rows)

    # breakdown points (first dose past half-rise in recon error)
    print("\n  breakdown points (recon-error half-rise):")
    for op in OPS:
        pts = [(s, e) for (o, s, e, a) in rows if o == op]
        es = [e for _, e in pts]; thr = min(es) + 0.5 * (max(es) - min(es))
        knee = next((s for s, e in pts if e >= thr), None)
        aucs = [a for (o, s, e, a) in rows if o == op]
        print(f"    {op:14} breakdown≈{knee}   chainAUC {aucs[0]:.2f}->{aucs[-1]:.2f} over dose")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for op in OPS:
            xs = [s for (o, s, e, a) in rows if o == op]
            ax1.plot(xs, [e for (o, s, e, a) in rows if o == op], marker="o", label=op)
            ax2.plot(xs, [a for (o, s, e, a) in rows if o == op], marker="o", label=op)
        ax1.set_title("reconstruction error vs dose"); ax1.set_xlabel("dose"); ax1.set_ylabel("1 - cos")
        ax2.set_title("chain A/B AUC vs dose"); ax2.set_xlabel("dose"); ax2.set_ylabel("AUC")
        ax2.axhline(0.5, color="grey", ls="--", lw=1)
        ax1.legend(fontsize=8); ax2.legend(fontsize=8); fig.tight_layout()
        fig.savefig(out / "dose_response_sdae.png", dpi=140)
        print(f"\n  saved {out/'dose_response_sdae.png'} and dose_response_sdae.csv")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()

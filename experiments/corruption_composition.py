"""
corruption_composition.py — #2: which corruption drives the representation?

Trains the frozen step-SDAE (denoise + PRM + chain) under different TRAINING corruption
compositions and measures each one's confound-controlled f1_B (multi-seed). Diagnostic: does
the denoising signal depend on a specific corruption, or do the supervised heads carry it
regardless (which the toxicology already hinted)?

Compositions (applied to the step-embedding INPUT; target is always the clean sequence):
  none     no corruption (pure AE + heads)
  mask     mask a fraction of step vectors (the default; forces filling-in from neighbours)
  shuffle  permute a fraction of step positions (breaks order)
  noise    additive Gaussian noise on step vectors
  all      mask + shuffle + noise together

    python experiments/corruption_composition.py --seeds 0,1,2 --epochs 20 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
from sdae_prm import StepSDAE_PRM, losses
from train_sdae import StepDS, collate, make_corrupt
from train_sdae_e2e import split_recs
from multiseed_ablation import build_confounds, probe_f1, pooled_reps

COMPS = ["none", "mask", "shuffle", "noise", "all"]


def apply_corruption(X, pad, comp, frac, gen, dev):
    """Return (corrupted_input, cm). Target stays the clean X."""
    Xc = X.clone(); cm = None
    if comp in ("mask", "all"):
        cm = make_corrupt(pad.cpu(), frac, gen).to(dev)
    if comp in ("noise", "all"):
        Xc = Xc + frac * Xc.std() * torch.randn(Xc.shape, generator=gen).to(dev)
    if comp in ("shuffle", "all"):
        for b in range(Xc.size(0)):
            n = int((~pad[b]).sum().item()); k = int(n * frac)
            if k >= 2:
                idx = torch.randperm(n, generator=gen)[:k]
                Xc[b, idx] = Xc[b, idx[torch.randperm(k, generator=gen)]]
    return Xc, cm


def train_one(tr, comp, seed, dev, epochs, frac=0.25):
    torch.manual_seed(seed)
    model = StepSDAE_PRM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(StepDS(tr), batch_size=32, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    for ep in range(epochs):
        model.train()
        for X, SL, pad, ch in loader:
            X, SL, pad, ch = X.to(dev), SL.to(dev), pad.to(dev), ch.to(dev)
            Xc, cm = apply_corruption(X, pad, comp, frac, gen, dev)
            recon, prm, cl, _ = model(Xc, pad, cm)
            tot, *_ = losses(recon, X, prm, SL, cl, ch, pad, cm, 1.0, 1.0)  # target = clean X
            opt.zero_grad(); tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "results_corruption"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in recs]); C = build_confounds(recs, args.data_dir)

    results = {c: [] for c in COMPS}
    for comp in COMPS:
        for s in seeds:
            tr, va = split_recs(recs, seed=s)   # same per-seed split as the multiseed run
            model = train_one(tr, comp, s, args.device, args.epochs)
            Z = pooled_reps(model, recs, args.device)
            f1 = probe_f1(Z, C, y, s)
            results[comp].append(f1)
            print(f"  {comp:8} seed {s}: f1_B={f1:.3f}", flush=True)

    import csv
    with (out / "corruption_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["composition"] + [f"seed{s}" for s in seeds] + ["mean", "std"])
        for c in COMPS:
            v = np.array(results[c]); w.writerow([c] + list(np.round(v, 4)) + [round(v.mean(), 4), round(v.std(), 4)])

    print("\n" + "=" * 50)
    print("  CORRUPTION COMPOSITION — confound-controlled f1_B")
    print("=" * 50)
    base = np.mean(results["mask"])
    for c in COMPS:
        v = np.array(results[c])
        print(f"  {c:8} : {v.mean():.3f} ± {v.std():.3f}   (Δ vs mask {v.mean()-base:+.3f})")
    print("=" * 50)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        data = [results[c] for c in COMPS]
        try: ax.boxplot(data, tick_labels=COMPS, showmeans=True)
        except TypeError: ax.boxplot(data, labels=COMPS, showmeans=True)
        for i, d in enumerate(data, 1):
            ax.scatter([i] * len(d), d, color="black", alpha=0.5, zorder=3)
        ax.set_ylabel("confound-controlled f1_B"); ax.set_title(f"Corruption composition ({len(seeds)} seeds)")
        fig.tight_layout(); fig.savefig(out / "corruption_boxplot.png", dpi=140)
        print(f"  saved {out/'corruption_boxplot.png'} and corruption_metrics.csv")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()

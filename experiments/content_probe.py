"""
content_probe.py — Phase 1c: the content-preservation probe.

Phase 1b showed the CLASSIFIER's f1_B is flat across corruption regimes -> the denoising
objective is inert *for classification*. That does not prove the latent z is empty of content:
a flat classifier score only proves the classifier doesn't NEED reconstruction. This probe
checks z directly, with the two supervised heads switched OFF, and asks: does z hold content
when there is zero loss competition?

Variant R (recon-only, --heads none): train new, L_denoise only, N seeds.
Variant F (full, the current model): the Phase-1a frozen full checkpoints (denoise+PRM+chain),
    same frozen embeddings and same per-seed splits as R -> a clean apples-to-apples control
    (one variable: heads on vs off). We deliberately do NOT use sdae_e2e_best.pt: it bundles a
    fine-tuned MiniLM, so its Transformer expects a different embedding distribution than the
    frozen step_cache -> not directly comparable.

Probe 1 (primary): nearest-neighbour retrieval. r_i = decode(encode(corrupted_i)); target
    t_i = clean step embeddings (the L_denoise target, a FROZEN quantity — the MiniLM step
    vectors are never trained, so co-adaptation collapse is impossible by construction).
    Candidate-level: mean-pool r_i and t_i over valid steps (N~340 val -> chance R@1 ~ 1/N).
Probe 2: decode specificity — mean pairwise cosine among r_i (collapse detector), with the
    targets t_i as a ceiling reference.

    python experiments/content_probe.py --seeds 0,1,2,3,4 --device cpu
"""
from __future__ import annotations

import argparse
import csv
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

F_CKPTS = ROOT / "experiments/results_multiseed/ckpts"   # Phase-1a frozen full checkpoints


def split_for_seed(recs, seed):
    """The exact 80/20 split train_sdae.py uses for a given seed (so R/F share it)."""
    rng = np.random.RandomState(seed); idx = np.arange(len(recs)); rng.shuffle(idx)
    cut = int(0.8 * len(recs))
    return [recs[i] for i in idx[:cut]], [recs[i] for i in idx[cut:]]


def train_variant_R(tr, va, seed, dev, epochs=30, patience=6, frac=0.25):
    """Recon-only training (L_denoise alone). Early-stop / checkpoint on best val L_denoise."""
    torch.manual_seed(seed)
    model = StepSDAE_PRM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    trl = DataLoader(StepDS(tr), batch_size=32, shuffle=True, collate_fn=collate,
                     generator=torch.Generator().manual_seed(seed))
    vloader = DataLoader(StepDS(va), batch_size=64, collate_fn=collate)
    best, best_state, no_imp, first_ld = 1e9, None, 0, None
    for ep in range(epochs):
        model.train()
        for X, SL, pad, ch in trl:
            X, SL, pad, ch = X.to(dev), SL.to(dev), pad.to(dev), ch.to(dev)
            cm = make_corrupt(pad.cpu(), frac, gen).to(dev)
            recon, prm, cl, _ = model(X, pad, cm)
            tot, ld, *_ = losses(recon, X, prm, SL, cl, ch, pad, cm, heads="none")
            opt.zero_grad(); tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val_ld = eval_denoise(model, vloader, dev, frac)
        if first_ld is None:
            first_ld = val_ld
        if val_ld < best - 1e-4:
            best, no_imp = val_ld, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, best, first_ld


def eval_denoise(model, loader, dev, frac):
    """Mean val L_denoise under the same masking corruption (fixed gen for comparability)."""
    model.eval(); gen = torch.Generator().manual_seed(0); tot = 0.0; n = 0
    with torch.no_grad():
        for X, SL, pad, ch in loader:
            X, pad = X.to(dev), pad.to(dev)
            cm = make_corrupt(pad.cpu(), frac, gen).to(dev)
            recon, _, _, _ = model(X, pad, cm)
            valid = ~pad
            cos = (F.normalize(recon, dim=-1) * F.normalize(X, dim=-1)).sum(-1)
            tot += float((1.0 - cos)[valid].sum()); n += int(valid.sum())
    return tot / max(n, 1)


def pooled_recon_and_target(model, va, dev, frac=0.25, corrupt_seed=1234):
    """Per-candidate mean-pooled reconstruction r_i (from corrupted input) and clean target t_i.

    Corruption uses a fixed generator so Variant R and Variant F see identical masks."""
    gen = torch.Generator().manual_seed(corrupt_seed)
    R, Tt = [], []
    model.eval()
    with torch.no_grad():
        for X, SL, pad, ch in DataLoader(StepDS(va), batch_size=64, collate_fn=collate,
                                         shuffle=False):
            X, pad = X.to(dev), pad.to(dev)
            cm = make_corrupt(pad.cpu(), frac, gen).to(dev)
            recon, _, _, _ = model(X, pad, cm)                  # (B,T,384)
            valid = (~pad).float().unsqueeze(-1)                # (B,T,1)
            denom = valid.sum(1).clamp(min=1)
            R.append(((recon * valid).sum(1) / denom).cpu().numpy())
            Tt.append(((X * valid).sum(1) / denom).cpu().numpy())
    return np.concatenate(R, 0), np.concatenate(Tt, 0)


def retrieval_metrics(Rp, Tp):
    """Rank each r_i against all targets t_j by cosine; return R@1, R@10, median rank, MRR."""
    r = Rp / (np.linalg.norm(Rp, axis=1, keepdims=True) + 1e-8)
    t = Tp / (np.linalg.norm(Tp, axis=1, keepdims=True) + 1e-8)
    S = r @ t.T                                                 # (N,N) sim(r_i, t_j)
    N = S.shape[0]
    order = np.argsort(-S, axis=1)                              # descending sim
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(N)])  # 1-indexed
    return dict(recall1=float((ranks == 1).mean()),
                recall10=float((ranks <= 10).mean()),
                median_rank=float(np.median(ranks)),
                mrr=float((1.0 / ranks).mean()),
                N=N)


def mean_pairwise_cos(M):
    m = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    S = m @ m.T
    N = S.shape[0]
    off = (S.sum() - np.trace(S)) / (N * (N - 1))               # exclude diagonal
    return float(off)


def run_variant(model, va, dev, tag):
    Rp, Tp = pooled_recon_and_target(model, va, dev)
    ret = retrieval_metrics(Rp, Tp)
    spec = mean_pairwise_cos(Rp)
    tgt_spec = mean_pairwise_cos(Tp)                            # ceiling reference
    r_std = float(Rp.std(0).mean())                             # sanity: non-zero variation
    print(f"  [{tag}] R@1={ret['recall1']:.3f} R@10={ret['recall10']:.3f} "
          f"medrank={ret['median_rank']:.0f} MRR={ret['mrr']:.3f} | "
          f"decode_spec={spec:.3f} (target ceil {tgt_spec:.3f}) r_std={r_std:.4f}", flush=True)
    return dict(**ret, decode_spec=spec, target_spec=tgt_spec, r_std=r_std)


def agg(rows, key):
    v = np.array([r[key] for r in rows]); return v.mean(), v.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "results_content_probe"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = torch.load(args.cache, weights_only=False)
    dev = torch.device(args.device)

    R_rows, F_rows = [], []
    for s in seeds:
        tr, va = split_for_seed(recs, s)
        # Variant R — recon-only, trained fresh on this seed's train split
        modelR, valld, first_ld = train_variant_R(tr, va, s, dev, epochs=args.epochs)
        print(f"seed {s}: Variant R val L_denoise {first_ld:.3f} -> {valld:.3f} "
              f"({'decreased OK' if valld < first_ld else 'FLAT/BROKEN'})", flush=True)
        rR = run_variant(modelR, va, dev, f"R s{s}"); rR["seed"] = s; rR["val_denoise"] = valld
        R_rows.append(rR)
        # Variant F — Phase-1a frozen full checkpoint for the SAME seed, SAME val split
        ckpt = F_CKPTS / f"frozen_seed{s}" / "sdae_best.pt"
        modelF = StepSDAE_PRM().to(dev)
        modelF.load_state_dict(torch.load(ckpt, map_location=dev)); modelF.eval()
        valld_F = eval_denoise(modelF, DataLoader(StepDS(va), batch_size=64, collate_fn=collate),
                               dev, 0.25)
        rF = run_variant(modelF, va, dev, f"F s{s}"); rF["seed"] = s; rF["val_denoise"] = valld_F
        F_rows.append(rF)

    keys = ["recall1", "recall10", "median_rank", "mrr", "decode_spec", "target_spec", "val_denoise"]
    with (out / "content_probe_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "seed"] + keys)
        for r in R_rows: w.writerow(["R", r["seed"]] + [round(r[k], 4) for k in keys])
        for r in F_rows: w.writerow(["F", r["seed"]] + [round(r[k], 4) for k in keys])

    print("\n" + "=" * 66)
    print("  PHASE 1c — CONTENT-PRESERVATION PROBE (mean ± std over seeds)")
    print(f"  chance R@1 ~ {1.0/R_rows[0]['N']:.4f}  (N={R_rows[0]['N']} val candidates)")
    print("=" * 66)
    hdr = f"  {'metric':<14}{'R (recon-only)':<22}{'F (full, frozen)':<22}"
    print(hdr); print("  " + "-" * 62)
    for k, label in [("recall1", "Recall@1"), ("recall10", "Recall@10"),
                     ("median_rank", "Median rank"), ("mrr", "MRR"),
                     ("decode_spec", "Decode spec"), ("val_denoise", "Val L_denoise")]:
        rm, rs = agg(R_rows, k); fm, fs = agg(F_rows, k)
        print(f"  {label:<14}{rm:.3f} ± {rs:.3f}       {fm:.3f} ± {fs:.3f}")
    tm, ts = agg(R_rows, "target_spec")
    print(f"  {'Target spec':<14}{tm:.3f} ± {ts:.3f}   (ceiling: decoded can't beat this)")
    print("=" * 66)
    print(f"  wrote {out/'content_probe_metrics.csv'}")


if __name__ == "__main__":
    main()

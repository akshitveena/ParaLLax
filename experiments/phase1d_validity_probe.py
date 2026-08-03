"""
phase1d_validity_probe.py — does the content-preserving latent still carry validity?

Phase 1c: content survives when the validity heads are removed (Variant R: Recall@1 0.996).
Phase 1d asks the other half: is any A/B validity signal STILL inside that content-preserving z,
even though nothing ever trained it to be there? Variant R never had a classifier — so this is a
held-out linear probe on frozen z, the identical confound-controlled protocol behind F's 0.497.

No new modelling. R is retrained deterministically (same seeds/config/splits as Phase 1c — CPU
seeded, so it reproduces the exact Phase-1c checkpoints), then frozen; clean candidates in, no
corruption, no grad. Probe = LogisticRegression on residualized z (length/latex/#steps/dataset
residualized out), fit on train split, eval on held-out val. Same readout as F (attention-pooled
step-code) so R and F drop into one table. Chance/floor ref: pooled raw-SBERT 0.291 ± 0.045.

    python experiments/phase1d_validity_probe.py --seeds 0,1,2,3,4 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
from sdae_prm import StepSDAE_PRM
from multiseed_ablation import build_confounds, pooled_reps
from content_probe import split_for_seed, train_variant_R

F_CKPTS = ROOT / "experiments/results_multiseed/ckpts"


def probe_full(Z, C, y, seed):
    """Confound-controlled, leakage-free A/B probe on the seed's own 80/20 split.
    Returns (f1_B, AUC, accuracy) — same residualize→fit-on-train→eval-on-val as Phase 1a."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    rng = np.random.RandomState(seed); idx = np.arange(len(Z)); rng.shuffle(idx)
    cut = int(0.8 * len(Z)); tri, vai = idx[:cut], idx[cut:]
    beta, *_ = np.linalg.lstsq(C[tri], Z[tri], rcond=None)      # residualizer fit on train only
    Zr = Z - C @ beta
    clf = LogisticRegression(max_iter=2000).fit(Zr[tri], y[tri])
    pred = clf.predict(Zr[vai]); proba = clf.predict_proba(Zr[vai])[:, list(clf.classes_).index("B")]
    yb = (y[vai] == "B").astype(int)
    return (float(f1_score(y[vai], pred, pos_label="B")),
            float(roc_auc_score(yb, proba)) if yb.sum() and (yb == 0).sum() else float("nan"),
            float(accuracy_score(y[vai], pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "results_content_probe"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in recs]); C = build_confounds(recs, args.data_dir)
    dev = torch.device(args.device)

    R_rows, F_rows = [], []
    for s in seeds:
        tr, va = split_for_seed(recs, s)
        # Variant R — recon-only, retrained deterministically (reproduces the Phase-1c checkpoint)
        modelR, _, _ = train_variant_R(tr, va, s, dev, epochs=args.epochs)
        ZR = pooled_reps(modelR, recs, dev)                     # clean, attention-pooled z
        f1R, aucR, accR = probe_full(ZR, C, y, s)
        R_rows.append((s, f1R, aucR, accR))
        # Variant F — Phase-1a frozen full checkpoint, identical protocol (fills AUC/acc for F)
        modelF = StepSDAE_PRM().to(dev)
        modelF.load_state_dict(torch.load(F_CKPTS / f"frozen_seed{s}" / "sdae_best.pt",
                                          map_location=dev)); modelF.eval()
        ZF = pooled_reps(modelF, recs, dev)
        f1F, aucF, accF = probe_full(ZF, C, y, s)
        F_rows.append((s, f1F, aucF, accF))
        print(f"  seed {s}:  R f1_B={f1R:.3f} AUC={aucR:.3f} acc={accR:.3f}   |   "
              f"F f1_B={f1F:.3f} AUC={aucF:.3f} acc={accF:.3f}", flush=True)

    with (out / "phase1d_validity_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["variant", "seed", "f1_B", "AUC", "accuracy"])
        for s, f1, a, ac in R_rows: w.writerow(["R", s, round(f1, 4), round(a, 4), round(ac, 4)])
        for s, f1, a, ac in F_rows: w.writerow(["F", s, round(f1, 4), round(a, 4), round(ac, 4)])

    def agg(rows, i): v = np.array([r[i] for r in rows]); return v.mean(), v.std()
    print("\n" + "=" * 60)
    print("  PHASE 1d — VALIDITY IN THE CONTENT-PRESERVING LATENT")
    print("  floor: pooled raw-SBERT 0.291 ± 0.045   |   F reference: 0.497 ± 0.047")
    print("=" * 60)
    for name, rows in [("R (recon-only)", R_rows), ("F (full)", F_rows)]:
        fm, fs = agg(rows, 1); am, a_s = agg(rows, 2); cm, cs = agg(rows, 3)
        print(f"  {name:<16} f1_B={fm:.3f} ± {fs:.3f}   AUC={am:.3f} ± {a_s:.3f}   "
              f"acc={cm:.3f} ± {cs:.3f}")
    print("=" * 60)
    print(f"  wrote {out/'phase1d_validity_metrics.csv'}")


if __name__ == "__main__":
    main()

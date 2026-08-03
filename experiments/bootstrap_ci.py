"""
bootstrap_ci.py — Phase 2e: bootstrap confidence intervals on headline numbers.

No retraining of anything that already has a checkpoint. For each headline representation we
recover the held-out predictions under the SAME confound-controlled protocol used everywhere
(residualize length/latex/#steps/dataset, fit on train, evaluate on held-out), then bootstrap
the held-out set 10,000x.

TWO tests are reported, because they answer different questions:
  * MARGINAL CI  — the uncertainty on each point estimate on its own.
  * PAIRED DIFFERENCE CI — the correct test for "is A better than B". Every representation
    shares the same val split for a given seed, so the same resample is applied to both and
    the difference is computed within-resample. Overlapping marginal CIs are a well-known
    OVERLY CONSERVATIVE heuristic (two intervals can overlap while the paired difference is
    reliably non-zero), so a claim is only softened if the DIFFERENCE interval contains 0.

Resampling is done at the PROBLEM level so all candidates of a problem move together. NOTE:
in this corpus that is a no-op — 1,700 records map to 1,700 unique problems (max 1 candidate
per problem), so problem-level degenerates to record-level. Implemented correctly anyway and
reported honestly rather than claimed as a safeguard we did not need.

Covered here: pooled raw-SBERT, frozen step-SDAE (F), Phase-1d recon-only latent (R). The e2e
rows (2a) and domain-transfer AUC are added once those checkpoints exist.

    python experiments/bootstrap_ci.py --seeds 0,1,2,3,4 --n_boot 10000 --include_R
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
import time
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
R_CKPTS = ROOT / "experiments/results_content_probe/ckpts"


def heldout_predictions(Z, C, y, seed):
    """Confound-controlled probe; return (y_val, pred_val, val_indices) on the seed's split."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(seed); idx = np.arange(len(Z)); rng.shuffle(idx)
    cut = int(0.8 * len(Z)); tri, vai = idx[:cut], idx[cut:]
    beta, *_ = np.linalg.lstsq(C[tri], Z[tri], rcond=None)      # fit on train only
    Zr = Z - C @ beta
    clf = LogisticRegression(max_iter=2000).fit(Zr[tri], y[tri])
    return y[vai], clf.predict(Zr[vai]), vai


def f1_B(y_true, y_pred):
    tp = int(((y_pred == "B") & (y_true == "B")).sum())
    fp = int(((y_pred == "B") & (y_true != "B")).sum())
    fn = int(((y_pred != "B") & (y_true == "B")).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def make_resamples(groups, n_boot, seed):
    """Shared bootstrap index sets, resampled over PROBLEM groups."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    member = [np.where(groups == g)[0] for g in uniq]
    picks = rng.randint(0, len(uniq), (n_boot, len(uniq)))
    return [np.concatenate([member[p] for p in row]) for row in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--include_R", action="store_true",
                    help="also cover the Phase-1d recon-only latent (reuses saved R ckpts)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=str(HERE / "results_bootstrap"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    R_CKPTS.mkdir(parents=True, exist_ok=True)
    recs = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in recs]); C = build_confounds(recs, args.data_dir)
    dev = torch.device(args.device)

    problem = np.array([re.sub(r"_c\d+$", "", r["id"]) for r in recs])
    n_prob, n_rec = len(set(problem)), len(recs)
    print(f"[2e] {n_rec} records / {n_prob} unique problems "
          f"({'problem-level == record-level (no-op)' if n_prob == n_rec else 'grouping active'})")

    sbert = np.array([r["steps_emb"].mean(0) for r in recs])

    def frozen_Z(s):
        m = StepSDAE_PRM().to(dev)
        m.load_state_dict(torch.load(F_CKPTS / f"frozen_seed{s}" / "sdae_best.pt", map_location=dev))
        m.eval(); return pooled_reps(m, recs, dev)

    def R_Z(s):
        ck = R_CKPTS / f"R_seed{s}.pt"
        m = StepSDAE_PRM().to(dev)
        if ck.exists():
            m.load_state_dict(torch.load(ck, map_location=dev))
        else:
            tr, va = split_for_seed(recs, s)
            m, _, _ = train_variant_R(tr, va, s, dev, epochs=args.epochs)
            torch.save(m.state_dict(), ck)
        m.eval(); return pooled_reps(m, recs, dev)

    reps = [("pooled raw-SBERT", lambda s: sbert), ("frozen step-SDAE (F)", frozen_Z)]
    if args.include_R:
        reps.append(("recon-only latent (R)", R_Z))

    # ---- collect held-out predictions (same val split per seed across all representations) ----
    preds = {name: {} for name, _ in reps}
    for name, Zf in reps:
        for s in seeds:
            preds[name][s] = heldout_predictions(Zf(s), C, y, s)

    # ---- one shared set of resamples per seed, reused by every representation ----
    boots = {}
    for s in seeds:
        _, _, vai = preds[reps[0][0]][s]
        boots[s] = make_resamples(problem[vai], args.n_boot, seed=s)

    # ---- marginal CIs ----
    curves = {name: {} for name, _ in reps}     # per-seed bootstrap f1 distributions
    summary = {}
    for name, _ in reps:
        pts, los, his = [], [], []
        for s in seeds:
            yv, pv, _ = preds[name][s]
            dist = np.array([f1_B(yv[b], pv[b]) for b in boots[s]])
            curves[name][s] = dist
            pt = f1_B(yv, pv)
            lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
            pts.append(pt); los.append(lo); his.append(hi)
            print(f"  {name:<22} seed {s}: f1_B={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]", flush=True)
        summary[name] = (float(np.mean(pts)), float(np.std(pts)),
                         float(np.mean(los)), float(np.mean(his)))

    print("\n" + "=" * 74)
    print(f"  PHASE 2e — BOOTSTRAP 95% CIs ({args.n_boot:,} resamples, problem-level)")
    print("=" * 74)
    for name, (m, sd, lo, hi) in summary.items():
        print(f"  {name:<22} {m:.3f} ± {sd:.3f}   marginal 95% CI [{lo:.3f}, {hi:.3f}]")

    # ---- paired difference CIs (the correct A-vs-B test) ----
    print("\n  PAIRED DIFFERENCE (same resample applied to both; 0 outside CI => real):")
    pairs = []
    for a, b in itertools.combinations([n for n, _ in reps], 2):
        d_pt, d_lo, d_hi = [], [], []
        for s in seeds:
            diff = curves[a][s] - curves[b][s]
            d_pt.append(float(np.mean(diff)))
            d_lo.append(float(np.percentile(diff, 2.5)))
            d_hi.append(float(np.percentile(diff, 97.5)))
        m, lo, hi = float(np.mean(d_pt)), float(np.mean(d_lo)), float(np.mean(d_hi))
        sig = not (lo <= 0.0 <= hi)
        marg_ov = not (summary[a][3] < summary[b][2] or summary[b][3] < summary[a][2])
        pairs.append((a, b, m, lo, hi, sig, marg_ov))
        print(f"    {a} − {b}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"=> {'SIGNIFICANT' if sig else 'not significant — SOFTEN claim'}"
              f"{'  (marginal CIs overlapped; paired test is the valid one)' if sig and marg_ov else ''}")
    print("=" * 74)

    with (out / "bootstrap_ci.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "a", "b", "value", "ci_lo", "ci_hi", "significant"])
        for name, (m, sd, lo, hi) in summary.items():
            w.writerow(["marginal", name, "", round(m, 4), round(lo, 4), round(hi, 4), ""])
        for a, b, m, lo, hi, sig, _ in pairs:
            w.writerow(["paired_diff", a, b, round(m, 4), round(lo, 4), round(hi, 4), int(sig)])

    print(f"  wall-clock {(time.time()-t0)/60:.1f} min | device={args.device} (Appendix A.4)")
    print(f"  wrote {out/'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()

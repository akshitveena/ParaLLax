"""
difficulty_baseline.py — W1: the difficulty-only null model the paper is missing.

The paper's thesis is that Type-B detectors secretly read DIFFICULTY. The obvious question —
how well does difficulty ALONE do? — is never answered. Raw-SBERT (0.291) is NOT that null
model. The null model is a classifier fit on nothing but the four confound variables
[length, latex_density, n_steps, dataset], no text. This script reports it, and every
step-structured / pooled number should be read against THIS, not against 0.291.

Bundled cheap wins from the same machinery (all writing-adjacent audit items):
  W12  base rate + AUPRC alongside f1_B (f1 is imbalance-sensitive)
  W13  each confound's individual correlation with the Type-B label (justifies the set)
  W2   stratified: f1_B WITHIN each source dataset — if difficulty-only collapses within a
       single dataset (where difficulty range is narrow), that is the cleanest possible proof
       the signal is difficulty; if the step-structured model still beats it within-dataset,
       that is the cleanest proof the signal is NOT only difficulty.

    python experiments/difficulty_baseline.py --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))
from multiseed_ablation import build_confounds, pooled_reps
from sdae_prm import StepSDAE_PRM

F_CKPTS = ROOT / "experiments/results_multiseed/ckpts"


def split(n, seed):
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    cut = int(0.8 * n); return idx[:cut], idx[cut:]


def scores(Xfeat, y, seed):
    """LogisticRegression on Xfeat; held-out f1_B / AUC / accuracy / AUPRC + base rate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
    tri, vai = split(len(y), seed)
    clf = LogisticRegression(max_iter=2000, class_weight=None).fit(Xfeat[tri], y[tri])
    proba = clf.predict_proba(Xfeat[vai])[:, list(clf.classes_).index("B")]
    pred = clf.predict(Xfeat[vai]); yb = (y[vai] == "B").astype(int)
    return dict(f1=f1_score(y[vai], pred, pos_label="B"),
                auc=roc_auc_score(yb, proba), acc=accuracy_score(y[vai], pred),
                auprc=average_precision_score(yb, proba), base=float(yb.mean()))


def agg(rows, k):
    v = np.array([r[k] for r in rows]); return v.mean(), v.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    recs = torch.load(args.cache, weights_only=False)
    y = np.array([r["chain"] for r in recs])
    C = build_confounds(recs, args.data_dir)          # [ones, len_z, latex_z, nsteps_z, ds_onehot]
    Cfeat = C[:, 1:]                                    # drop intercept; LR adds its own
    base = float((y == "B").mean())

    # ---- W1: difficulty-only (4 confounds, no text) ----
    diff_rows = [scores(Cfeat, y, s) for s in seeds]
    # reference: raw-SBERT pooled (the number the paper currently benchmarks against)
    sbert = np.array([r["steps_emb"].mean(0) for r in recs])
    sbert_rows = [scores(sbert, y, s) for s in seeds]
    # reference: frozen step-SDAE pooled rep (needs the Phase-1a ckpts)
    step_rows = []
    for s in seeds:
        ck = F_CKPTS / f"frozen_seed{s}" / "sdae_best.pt"
        if not ck.exists():
            step_rows = None; break
        m = StepSDAE_PRM(); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
        step_rows.append(scores(pooled_reps(m, recs, args.device), y, s))

    print("=" * 74)
    print(f"  W1 — DIFFICULTY-ONLY NULL MODEL   (base rate {base:.3f} Type-B, n={len(recs)})")
    print("=" * 74)
    print(f"  {'representation':<34}{'f1_B':<14}{'AUC':<14}{'AUPRC'}")
    print("  " + "-" * 70)

    def line(name, rows):
        fm, fs = agg(rows, "f1"); am, a_s = agg(rows, "auc"); pm, ps = agg(rows, "auprc")
        print(f"  {name:<34}{fm:.3f} ± {fs:.3f}  {am:.3f} ± {a_s:.3f}  {pm:.3f} ± {ps:.3f}")

    line("difficulty-only (4 confounds)", diff_rows)
    line("raw-SBERT (pooled, no control)", sbert_rows)
    if step_rows:
        line("frozen step-SDAE (no control)", step_rows)
    print(f"  (AUPRC chance = base rate = {base:.3f})")
    print("=" * 74)
    dm, _ = agg(diff_rows, "f1")
    print(f"  READING: difficulty alone reaches f1_B = {dm:.3f}. Every uncontrolled representation")
    print(f"  number must be read against THIS, not against raw-SBERT 0.291. If uncontrolled")
    print(f"  detectors sit near {dm:.3f}, difficulty explains most of their apparent skill.")
    print("=" * 74)

    # ---- W13: each confound's individual correlation with the Type-B label ----
    from scipy.stats import pointbiserialr
    yb = (y == "B").astype(int)
    names = ["length", "latex_density", "n_steps"]
    print("\n  W13 — individual confound correlation with Type-B label (point-biserial r):")
    for i, nm in enumerate(names):
        r, p = pointbiserialr(yb, C[:, 1 + i])
        print(f"    {nm:<16} r = {r:+.3f}  (p={p:.1e})")

    # ---- W2: stratified f1_B WITHIN each source dataset ----
    print("\n  W2 — difficulty-only vs step-structured, WITHIN each dataset (seed-avg f1_B):")
    print(f"    {'dataset':<16}{'n':>6}{'%B':>7}   {'diff-only':>10}{'step-SDAE':>12}")
    dsets = sorted(set(r["split"] for r in recs))
    for ds in dsets:
        mask = np.array([r["split"] == ds for r in recs])
        if mask.sum() < 40 or len(set(y[mask])) < 2:
            print(f"    {ds:<16}{int(mask.sum()):>6}   (too few / single-class — skipped)")
            continue
        yd = y[mask]; Cd = Cfeat[mask]
        d_f1, s_f1 = [], []
        for s in seeds:
            d_f1.append(scores(Cd, yd, s)["f1"])
            if step_rows:
                m = StepSDAE_PRM()
                m.load_state_dict(torch.load(F_CKPTS / f"frozen_seed{s}" / "sdae_best.pt",
                                             map_location="cpu")); m.eval()
                Zall = pooled_reps(m, recs, args.device)
                s_f1.append(scores(Zall[mask], yd, s)["f1"])
        sv = f"{np.mean(s_f1):.3f}" if s_f1 else "n/a"
        print(f"    {ds:<16}{int(mask.sum()):>6}{100*np.mean(yd=='B'):>6.1f}%   "
              f"{np.mean(d_f1):>10.3f}{sv:>12}")
    print("\n  If step-SDAE > diff-only WITHIN datasets, the signal is not only difficulty.")


if __name__ == "__main__":
    main()

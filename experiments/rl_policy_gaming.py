"""
rl_policy_gaming.py — E-RL: Closed-Loop Policy Gaming (Best-of-N).

The paper claims a policy optimized against an uncontrolled PRM faces a gradient
toward manipulating apparent difficulty. This experiment demonstrates it concretely:

  Best-of-N selection with two scorers on the SAME candidate pool:
    (a) UNCONTROLLED scorer  — LogReg on raw SBERT embeddings (confound-laden)
    (b) CONTROLLED scorer    — LogReg on linearly-residualized embeddings

For each problem, both scorers pick their top-1 candidate from N options. We then
measure the SURFACE STATISTICS (length, latex_density, n_steps) of the selected
candidates. If the uncontrolled scorer systematically picks longer, more LaTeX-heavy
responses while the controlled scorer does not, that is a direct demonstration of
policy gaming: the uncontrolled signal rewards difficulty theatre, not validity.

This is the smoking gun the paper currently gestures at but does not fire.

    python experiments/rl_policy_gaming.py --seeds 0,1,2,3,4
    python experiments/rl_policy_gaming.py --N 8 --seeds 0,1,2

Pre-requisites:
    data/step_cache.pt           (from build_step_embeddings.py)
    data/processed_pb/           (ProcessBench processed data)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "main")); sys.path.insert(0, str(HERE))

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from multiseed_ablation import build_confounds


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _latex_density(text: str) -> float:
    """Fraction of tokens that are LaTeX-like (backslash commands, braces, etc.)."""
    tokens = text.split()
    if not tokens:
        return 0.0
    latex_toks = sum(1 for t in tokens if "\\" in t or t in ("{", "}", "^", "_")
                     or t.startswith("\\") or "$" in t)
    return latex_toks / len(tokens)


def _n_steps(text: str) -> int:
    """Count reasoning steps (lines starting with a digit or bullet)."""
    import re
    return max(1, len(re.findall(r"(?m)^\s*(?:\d+[\.\):]|[-•])\s", text)))


def _length(text: str) -> int:
    return len(text.split())


def residualize_linear(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """OLS residualization: Z_resid = Z - C @ (C^T C)^{-1} C^T Z."""
    CtC_inv = np.linalg.pinv(C.T @ C)
    projection = C @ CtC_inv @ C.T @ Z
    return Z - projection


def split(n: int, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


# --------------------------------------------------------------------------- #
# Best-of-N Selection
# --------------------------------------------------------------------------- #
def best_of_n_experiment(recs, C, N: int, seed: int):
    """Run Best-of-N with controlled vs uncontrolled scorers.

    For each 'problem group' (consecutive N records), each scorer picks its
    top-1 candidate. We measure surface stats of the selections.
    """
    y = np.array([r["chain"] for r in recs])
    # Raw SBERT pooled embeddings (uncontrolled)
    Z_raw = np.array([r["steps_emb"].mean(0) for r in recs])
    # Residualized embeddings (controlled)
    Cfeat = C[:, 1:]  # drop intercept
    Z_ctrl = residualize_linear(Z_raw, C)

    # Train both scorers on a held-out split
    tri, vai = split(len(y), seed)
    yb = (y == "B").astype(int)

    clf_raw = LogisticRegression(max_iter=2000).fit(Z_raw[tri], yb[tri])
    clf_ctrl = LogisticRegression(max_iter=2000).fit(Z_ctrl[tri], yb[tri])

    # Use validation set for Best-of-N simulation
    val_idx = vai
    if len(val_idx) < N:
        return None

    # Group validation candidates into pseudo-problems of size N
    rng = np.random.RandomState(seed + 100)
    rng.shuffle(val_idx)
    n_groups = len(val_idx) // N

    raw_selected_stats = {"length": [], "latex": [], "steps": []}
    ctrl_selected_stats = {"length": [], "latex": [], "steps": []}
    random_stats = {"length": [], "latex": [], "steps": []}

    for g in range(n_groups):
        group_idx = val_idx[g * N: (g + 1) * N]

        # Score each candidate in the group
        Z_raw_group = Z_raw[group_idx]
        Z_ctrl_group = Z_ctrl[group_idx]

        # P(Type-B) — higher = model thinks it's more "valid-looking"
        # We want the scorer to PREFER candidates: pick highest P(not-B) = lowest P(B)
        raw_scores = clf_raw.predict_proba(Z_raw_group)[:, 0]   # P(not-B) = P(good)
        ctrl_scores = clf_ctrl.predict_proba(Z_ctrl_group)[:, 0]

        raw_pick = group_idx[np.argmax(raw_scores)]
        ctrl_pick = group_idx[np.argmax(ctrl_scores)]
        rand_pick = group_idx[rng.randint(0, N)]

        # Measure surface stats of the picked candidate
        for pick, stats in [(raw_pick, raw_selected_stats),
                            (ctrl_pick, ctrl_selected_stats),
                            (rand_pick, random_stats)]:
            rec = recs[pick]
            text = rec.get("full_text", "")
            if not text:
                # reconstruct from steps
                text = " ".join(s if isinstance(s, str) else str(s)
                                for s in rec.get("steps_raw", rec.get("steps_text", [""])))
            stats["length"].append(_length(text))
            stats["latex"].append(_latex_density(text))
            stats["steps"].append(_n_steps(text))

    return {
        "n_groups": n_groups,
        "raw": {k: (np.mean(v), np.std(v)) for k, v in raw_selected_stats.items()},
        "ctrl": {k: (np.mean(v), np.std(v)) for k, v in ctrl_selected_stats.items()},
        "random": {k: (np.mean(v), np.std(v)) for k, v in random_stats.items()},
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="E-RL: Best-of-N policy gaming — uncontrolled vs controlled scorer")
    ap.add_argument("--cache", default=str(ROOT / "data/step_cache.pt"))
    ap.add_argument("--data_dir", default=str(ROOT / "data/processed_pb"))
    ap.add_argument("--N", type=int, default=4, help="candidates per pseudo-problem")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    recs = torch.load(args.cache, weights_only=False)
    C = build_confounds(recs, args.data_dir)
    y = np.array([r["chain"] for r in recs])
    base_b = float((y == "B").mean())

    print("=" * 78)
    print(f"  E-RL — CLOSED-LOOP POLICY GAMING (Best-of-{args.N})")
    print(f"  n={len(recs)}  base_rate_B={base_b:.3f}  seeds={seeds}")
    print("=" * 78)

    # Aggregate across seeds
    all_results = []
    for s in seeds:
        res = best_of_n_experiment(recs, C, args.N, s)
        if res:
            all_results.append(res)

    if not all_results:
        print("  ERROR: not enough data for Best-of-N grouping."); return

    # Print results table
    print(f"\n  Surface statistics of Best-of-{args.N} selected candidates (mean ± std across seeds):\n")
    print(f"  {'scorer':<28}{'length':>14}{'latex_density':>16}{'n_steps':>12}")
    print("  " + "-" * 70)

    for label, key in [("Random baseline", "random"),
                       ("Uncontrolled (raw SBERT)", "raw"),
                       ("Controlled (residualized)", "ctrl")]:
        lengths = [r[key]["length"][0] for r in all_results]
        latexes = [r[key]["latex"][0] for r in all_results]
        steps = [r[key]["steps"][0] for r in all_results]
        print(f"  {label:<28}"
              f"{np.mean(lengths):>8.1f} ± {np.std(lengths):>4.1f}"
              f"{np.mean(latexes):>10.4f} ± {np.std(latexes):>.4f}"
              f"{np.mean(steps):>7.1f} ± {np.std(steps):>.1f}")

    print()
    print("=" * 78)

    # Compute the gaming delta
    raw_len = np.mean([r["raw"]["length"][0] for r in all_results])
    ctrl_len = np.mean([r["ctrl"]["length"][0] for r in all_results])
    rand_len = np.mean([r["random"]["length"][0] for r in all_results])
    raw_latex = np.mean([r["raw"]["latex"][0] for r in all_results])
    ctrl_latex = np.mean([r["ctrl"]["latex"][0] for r in all_results])

    print(f"  GAMING DELTA (uncontrolled vs random):")
    print(f"    length:  {raw_len - rand_len:+.1f} tokens  "
          f"({'longer' if raw_len > rand_len else 'shorter'} = "
          f"{'GAMING DETECTED' if raw_len > rand_len * 1.05 else 'no significant gaming'})")
    print(f"    latex:   {raw_latex - ctrl_latex:+.4f}  "
          f"({'denser' if raw_latex > ctrl_latex else 'sparser'} = "
          f"{'GAMING DETECTED' if raw_latex > ctrl_latex * 1.05 else 'no significant gaming'})")
    print()
    print(f"  READING: If the uncontrolled scorer selects systematically LONGER and more")
    print(f"  LaTeX-dense candidates than the controlled scorer, it proves that optimizing")
    print(f"  against an uncontrolled PRM rewards difficulty theatre — not genuine validity.")
    print(f"  The confound-controlled scorer suppresses this exploit channel.")
    print("=" * 78)


if __name__ == "__main__":
    main()
